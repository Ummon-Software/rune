//! Sound for a clip on Linux: a `cpal` output stream on PipeWire/PulseAudio (with ALSA
//! fallback), fed by the FFmpeg decoder and timed by the player's clock. Each audio callback
//! writes the samples for the moment they'll be heard, padding with silence or skipping ahead
//! when the clock moves, so picture and sound stay in sync without blocking either thread.

use super::machine::Anchor;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Host, HostId, SampleFormat, SizedSample, Stream, StreamConfig};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Sound decoded ahead of what's heard; the decoder waits beyond it.
const AHEAD: f64 = 1.0;
/// How far written sound may drift from the clock before it jumps back in line.
const SLOP: f64 = 0.04;
/// Fallback output latency when the audio host doesn't report playback timestamps.
const DEFAULT_LATENCY: Duration = Duration::from_millis(20);

/// Negotiated device audio format for FFmpeg's resampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OutputSpec {
    pub channels: u16,
    pub rate: u32,
}

/// The playback clock the UI sets and sound follows: none while paused, stalled or ended.
#[derive(Default)]
pub(super) struct Clock {
    anchor: Mutex<Option<Anchor>>,
    changed: Condvar,
}

impl Clock {
    pub(super) fn set(&self, anchor: Option<Anchor>) {
        *self.anchor.lock().unwrap_or_else(|e| e.into_inner()) = anchor;
        self.changed.notify_all();
    }

    pub(super) fn get(&self) -> Option<Anchor> {
        *self.anchor.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait_change(&self, expected_running: bool, patience: Duration) {
        let guard = self.anchor.lock().unwrap_or_else(|e| e.into_inner());
        let running = guard.is_some_and(|a| (a.rate - 1.0).abs() < 1e-6);
        if running == expected_running {
            self.changed.wait_timeout(guard, patience).ok();
        }
    }
}

struct Chunk {
    pts: f64,
    samples: Vec<f32>,
}

#[derive(Default)]
struct Feed {
    chunks: VecDeque<Chunk>,
    /// Interleaved samples buffered.
    samples: usize,
    gain: f32,
    stop: bool,
    /// Media time of the next sample the device gets, once it's writing in line with the clock.
    cursor: Option<f64>,
}

/// A clip's sound output; dropping it stops the stream.
pub(super) struct Output {
    feed: Arc<Mutex<Feed>>,
    clock: Arc<Clock>,
    spec: OutputSpec,
}

impl Output {
    pub(super) fn open(clock: Arc<Clock>) -> Result<Self, String> {
        let feed = Arc::new(Mutex::new(Feed { gain: 1.0, ..Feed::default() }));
        let (thread_feed, thread_clock) = (feed.clone(), clock.clone());
        let (ready, started) = std::sync::mpsc::sync_channel::<Result<OutputSpec, String>>(1);
        std::thread::Builder::new()
            .name("rune-video-sound".into())
            .spawn(move || match open_stream(thread_feed.clone(), thread_clock.clone()) {
                Ok((stream, spec)) => {
                    ready.send(Ok(spec)).ok();
                    drive_stream(stream, &thread_feed, &thread_clock);
                }
                Err(e) => {
                    ready.send(Err(e)).ok();
                }
            })
            .map_err(|e| e.to_string())?;
        let spec = started.recv().map_err(|_| "the sound thread stopped".to_string())??;
        Ok(Self { feed, clock, spec })
    }

    pub(super) fn spec(&self) -> OutputSpec {
        self.spec
    }

    /// Queues interleaved float samples at `pts` seconds; sound before `skip_before` is dropped.
    pub(super) fn push(&self, samples: Vec<f32>, pts: f64, skip_before: f64) {
        let channels = usize::from(self.spec.channels).max(1);
        let rate = f64::from(self.spec.rate);
        let end = pts + (samples.len() / channels) as f64 / rate;
        if samples.is_empty() || end <= skip_before {
            return;
        }
        let mut feed = self.feed.lock().unwrap_or_else(|e| e.into_inner());
        feed.samples += samples.len();
        feed.chunks.push_back(Chunk { pts, samples });
    }

    /// Forgets queued sound (a seek).
    pub(super) fn flush(&self) {
        let mut feed = self.feed.lock().unwrap_or_else(|e| e.into_inner());
        feed.chunks.clear();
        feed.samples = 0;
        feed.cursor = None;
    }

    pub(super) fn set_gain(&self, gain: f32) {
        self.feed.lock().unwrap_or_else(|e| e.into_inner()).gain = gain.clamp(0.0, 1.0);
    }

    /// Enough is queued; the decoder may wait.
    pub(super) fn full(&self) -> bool {
        let per_second = f64::from(self.spec.rate) * f64::from(self.spec.channels);
        self.feed.lock().unwrap_or_else(|e| e.into_inner()).samples as f64 >= AHEAD * per_second
    }

    /// Nothing plays: paused, stalled or ended.
    pub(super) fn idle(&self) -> bool {
        self.clock.get().is_none()
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        self.feed.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        self.clock.changed.notify_all();
    }
}

/// Prefers PulseAudio (which connects natively to PipeWire via pipewire-pulse as well as
/// PulseAudio), falling back to the default ALSA host.
fn pick_host() -> Host {
    if let Ok(host) = cpal::host_from_id(HostId::PulseAudio)
        && host.default_output_device().is_some()
    {
        return host;
    }
    cpal::default_host()
}

fn open_stream(feed: Arc<Mutex<Feed>>, clock: Arc<Clock>) -> Result<(Stream, OutputSpec), String> {
    let host = pick_host();
    let device = host.default_output_device().ok_or("no sound output device")?;
    let supported = device
        .default_output_config()
        .map_err(|e| format!("no default sound output config: {e}"))?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let spec = OutputSpec { channels: config.channels.clamp(1, 8), rate: config.sample_rate };
    let config = StreamConfig { channels: spec.channels, ..config };
    let stream = match sample_format {
        SampleFormat::F32 => build_typed_stream::<f32>(&device, config, spec, feed, clock)?,
        SampleFormat::I16 => build_typed_stream::<i16>(&device, config, spec, feed, clock)?,
        SampleFormat::U16 => build_typed_stream::<u16>(&device, config, spec, feed, clock)?,
        SampleFormat::I32 => build_typed_stream::<i32>(&device, config, spec, feed, clock)?,
        other => return Err(format!("unsupported audio output sample format: {other:?}")),
    };
    stream.pause().ok();
    Ok((stream, spec))
}

fn build_typed_stream<T>(
    device: &cpal::Device,
    config: StreamConfig,
    spec: OutputSpec,
    feed: Arc<Mutex<Feed>>,
    clock: Arc<Clock>,
) -> Result<Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = usize::from(spec.channels);
    let rate = f64::from(spec.rate);
    let mut scratch = Vec::<f32>::new();
    device
        .build_output_stream(
            config,
            move |out: &mut [T], info: &cpal::OutputCallbackInfo| {
                let anchor = clock.get().filter(|a| (a.rate - 1.0).abs() < 1e-6);
                let Some(anchor) = anchor else {
                    out.fill(T::EQUILIBRIUM);
                    feed.lock().unwrap_or_else(|e| e.into_inner()).cursor = None;
                    return;
                };
                let raw = info.timestamp().playback.duration_since(info.timestamp().callback);
                let latency = if raw > Duration::ZERO && raw <= Duration::from_millis(250) {
                    raw
                } else {
                    DEFAULT_LATENCY
                };
                let heard = rune_core::time::now() + latency;
                scratch.resize(out.len(), 0.0);
                {
                    let mut guard = feed.lock().unwrap_or_else(|e| e.into_inner());
                    fill(&mut guard, &mut scratch, channels, rate, anchor.position(heard));
                }
                for (dst, &src) in out.iter_mut().zip(&scratch) {
                    *dst = T::from_sample(src.clamp(-1.0, 1.0));
                }
            },
            |err| eprintln!("rune: audio output error: {err}"),
            None,
        )
        .map_err(|e| format!("open sound output stream: {e}"))
}

fn drive_stream(stream: Stream, feed: &Mutex<Feed>, clock: &Clock) {
    let mut playing = false;
    loop {
        if feed.lock().unwrap_or_else(|e| e.into_inner()).stop {
            break;
        }
        let want_play = clock.get().is_some_and(|a| (a.rate - 1.0).abs() < 1e-6);
        if want_play && !playing {
            if stream.play().is_ok() {
                playing = true;
            }
        } else if !want_play && playing {
            stream.pause().ok();
            playing = false;
            feed.lock().unwrap_or_else(|e| e.into_inner()).cursor = None;
        }
        clock.wait_change(playing, Duration::from_millis(250));
    }
    stream.pause().ok();
}

/// Writes `out` with the sound from media time `want` on, in line with the clock.
fn fill(feed: &mut Feed, out: &mut [f32], channels: usize, rate: f64, want: f64) {
    let mut cursor = match feed.cursor {
        Some(c) if (c - want).abs() < SLOP => c,
        _ => want,
    };
    let frames = out.len() / channels;
    let mut written = 0;
    while written < frames {
        while let Some(c) = feed.chunks.front() {
            let end = c.pts + (c.samples.len() / channels) as f64 / rate;
            if end > cursor + 0.5 / rate {
                break;
            }
            feed.samples -= c.samples.len();
            feed.chunks.pop_front();
        }
        let Some(chunk) = feed.chunks.front() else {
            out[written * channels..].fill(0.0);
            cursor += (frames - written) as f64 / rate;
            break;
        };
        let start = ((cursor - chunk.pts) * rate).round() as i64;
        let n = if start < 0 {
            let n = ((-start) as usize).min(frames - written);
            out[written * channels..(written + n) * channels].fill(0.0);
            n
        } else {
            let start = start as usize;
            let available = (chunk.samples.len() / channels).saturating_sub(start);
            if available == 0 {
                if let Some(c) = feed.chunks.pop_front() {
                    feed.samples -= c.samples.len();
                }
                continue;
            }
            let n = available.min(frames - written);
            let gain = feed.gain;
            for (o, s) in out[written * channels..(written + n) * channels]
                .iter_mut()
                .zip(&chunk.samples[start * channels..])
            {
                *o = s * gain;
            }
            n
        };
        written += n;
        cursor += n as f64 / rate;
    }
    feed.cursor = Some(cursor);
}
