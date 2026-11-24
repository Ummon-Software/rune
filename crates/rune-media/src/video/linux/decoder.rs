//! One clip's decoder thread on Linux: opens the source with FFmpeg, decodes a few pictures
//! ahead, hands audio to the `cpal` sound output, loops without a gap by seeking back while
//! the last pictures wait in the queue, and parks after a few seconds undrawn.

use super::super::{Kind, MediaSource, Meta};
use super::audio::{self, Output};
use super::ffmpeg::{DecodedItem, Opened, Picture};
use crate::native_surface::ByteFrame;
use rune_core::time::Instant;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// Pictures decoded ahead of the one on screen.
const AHEAD: usize = 3;
/// A decoder no frame has taken a picture from for this long lets go of its source and decoders;
/// the clip keeps its last picture and reopens when drawn again.
const PARK: Duration = Duration::from_secs(3);
/// How often a decoder with a full audio buffer checks whether room freed up.
const SOUND_POLL: Duration = Duration::from_millis(5);

/// What the UI asks of the decoder and what the decoder hands back, under one lock.
#[derive(Default)]
pub(super) struct Inbox {
    /// A seek to land: clip seconds and the generation its pictures carry.
    pub seek: Option<(f64, u32)>,
    pub looping: bool,
    /// Play sound, at this gain (unmuted, and the clip has some).
    pub sound: Option<f32>,
    pub stop: bool,
    /// When a frame last drew the clip.
    pub touched: Option<Instant>,
    pub news: Vec<News>,
    pub frames: VecDeque<Decoded>,
    /// The generation whose last picture is in the queue: the clip ended there.
    pub ended: Option<u32>,
    /// Audio-only: the unwrapped media time of the last decoded audio sample, for end detection.
    pub audio_end: Option<f64>,
}

pub(super) enum News {
    Opened(Facts),
    Failed(String),
    Parked,
}

/// What opening found.
#[derive(Clone, Copy, Debug)]
pub(super) struct Facts {
    pub meta: Meta,
    pub picture: Option<Picture>,
    pub has_sound: bool,
}

pub(super) struct Decoded {
    pub generation: u32,
    /// Seconds from the clip's start, counting on through loops.
    pub pts: f64,
    pub frame: ByteFrame,
}

pub(super) struct Pipe {
    inbox: Mutex<Inbox>,
    wake: Condvar,
}

impl Pipe {
    pub(super) fn lock(&self) -> MutexGuard<'_, Inbox> {
        self.inbox.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn poke(&self) {
        self.wake.notify_all();
    }
}

pub(super) struct Job {
    pub source: MediaSource,
    pub headers: Vec<(String, String)>,
    pub kind: Kind,
    pub start: f64,
    pub generation: u32,
    pub looping: bool,
    pub sound: Option<f32>,
    pub notify: Box<dyn Fn() + Send>,
    pub clock: Arc<audio::Clock>,
}

/// Starts a decoder thread; it runs until `stop` or parking.
pub(super) fn start(job: Job) -> Arc<Pipe> {
    let pipe = Arc::new(Pipe {
        inbox: Mutex::new(Inbox {
            looping: job.looping,
            sound: job.sound,
            touched: Some(rune_core::time::now()),
            ..Inbox::default()
        }),
        wake: Condvar::new(),
    });
    let thread_pipe = pipe.clone();
    let spawned = std::thread::Builder::new()
        .name("rune-video-decode".into())
        .spawn(move || run(&thread_pipe, job));
    if spawned.is_err() {
        pipe.lock().news.push(News::Failed("couldn't start a video decoder thread".into()));
    }
    pipe
}

fn run(pipe: &Pipe, job: Job) {
    let mut d = match Decoder::open(&job) {
        Ok(d) => d,
        Err(e) => {
            pipe.lock().news.push(News::Failed(e));
            (job.notify)();
            return;
        }
    };
    pipe.lock().news.push(News::Opened(d.facts));
    (job.notify)();
    if let Err(e) = d.decode(pipe, &job) {
        pipe.lock().news.push(News::Failed(e));
        (job.notify)();
    }
}

struct Decoder {
    opened: Opened,
    facts: Facts,
    output: Option<Output>,
    audio_failed: bool,
}

impl Decoder {
    fn open(job: &Job) -> Result<Self, String> {
        let mut opened = Opened::open(&job.source, &job.headers, job.kind, true)?;
        if job.start > 0.0 {
            opened.seek(job.start)?;
        }
        let meta = Meta {
            duration: opened.duration,
            size: opened.picture.map(|p| p.natural),
            frame: opened.picture.map_or(1.0 / 30.0, |p| p.frame),
            sound: opened.has_sound,
        };
        let facts = Facts { meta, picture: opened.picture, has_sound: opened.has_sound };
        let mut d = Self { opened, facts, output: None, audio_failed: false };
        d.select_sound(job.sound.is_some() || job.kind == Kind::Audio, &job.clock);
        Ok(d)
    }

    fn select_sound(&mut self, on: bool, clock: &Arc<audio::Clock>) {
        if !self.facts.has_sound {
            return;
        }
        if on && self.output.is_none() && !self.audio_failed {
            match Output::open(clock.clone()) {
                Ok(output) => {
                    self.opened.set_audio_output(Some(output.spec()));
                    self.output = Some(output);
                }
                Err(e) => {
                    eprintln!("rune: no sound for the clip: {e}");
                    self.audio_failed = true;
                    self.opened.set_audio_output(None);
                }
            }
        }
        if !on {
            self.output = None;
            self.audio_failed = false;
            self.opened.set_audio_output(None);
        }
    }

    fn decode(&mut self, pipe: &Pipe, job: &Job) -> Result<(), String> {
        let mut generation = job.generation;
        let mut skip_before = job.start;
        let mut epoch = 0.0;
        let mut waiting_since = rune_core::time::now();
        let mut end_due: Option<f64> = None;
        let mut last_audio_end: f64 = job.start;
        let mut ever_decoded = false;
        let mut sent_video = false;
        let mut last_skipped: Option<(f64, ByteFrame)> = None;
        let mut audio_wait = false;

        loop {
            let decode_video;
            {
                let mut s = pipe.lock();
                loop {
                    if s.stop {
                        return Ok(());
                    }
                    if let Some((to, g)) = s.seek.take() {
                        end_due = None;
                        s.frames.clear();
                        s.ended = None;
                        s.audio_end = None;
                        generation = g;
                        skip_before = to;
                        epoch = 0.0;
                        last_audio_end = to;
                        sent_video = false;
                        last_skipped = None;
                        audio_wait = false;
                        self.opened.seek(to)?;
                        if let Some(o) = &self.output {
                            o.flush();
                        }
                    }
                    let want_sound = s.sound.is_some();
                    let sound_change = (want_sound && self.output.is_none() && !self.audio_failed)
                        || (!want_sound && (self.output.is_some() || self.audio_failed));
                    if sound_change {
                        let had_output = self.output.is_some();
                        audio_wait = false;
                        drop(s);
                        self.select_sound(want_sound, &job.clock);
                        s = pipe.lock();
                        if !had_output && self.output.is_some() && job.kind == Kind::Audio {
                            let cur = job
                                .clock
                                .get()
                                .map(|c| c.position(rune_core::time::now()).max(0.0));
                            let pos = cur.unwrap_or(skip_before);
                            let dur = self.facts.meta.duration;
                            let to = if dur > 0.0 && s.looping {
                                pos.rem_euclid(dur)
                            } else if dur > 0.0 {
                                pos.min(dur)
                            } else {
                                pos
                            };
                            epoch = if dur > 0.0 && s.looping {
                                (pos / dur).floor() * dur
                            } else {
                                0.0
                            };
                            skip_before = to;
                            last_audio_end = pos;
                            end_due = None;
                            s.ended = None;
                            s.audio_end = None;
                            self.opened.seek(to)?;
                            if let Some(o) = &self.output {
                                o.flush();
                            }
                        }
                    }
                    if let (Some(o), Some(gain)) = (&self.output, s.sound) {
                        o.set_gain(gain);
                    }
                    let has_video = self.facts.picture.is_some();
                    let silent_audio_ready =
                        !has_video && self.output.is_none() && ever_decoded;
                    if silent_audio_ready && !s.looping && s.ended != Some(generation) {
                        let dur_end = epoch + self.facts.meta.duration;
                        s.ended = Some(generation);
                        s.audio_end = Some(dur_end);
                        end_due = Some(dur_end);
                    }
                    let ended = s.ended == Some(generation);
                    let room = s.frames.len() < AHEAD;
                    let sound_full = self.output.as_ref().is_some_and(Output::full);
                    let need_sound = self.output.is_some() && !sound_full && !audio_wait;
                    if !ended
                        && ((has_video && room)
                            || (!has_video && !sound_full && !silent_audio_ready)
                            || need_sound)
                    {
                        decode_video = has_video && room;
                        if decode_video {
                            audio_wait = false;
                        }
                        break;
                    }
                    let touched = s.touched.unwrap_or(waiting_since);
                    let playing_sound = self.output.as_ref().is_some_and(|o| !o.idle());
                    let heard_end = end_due.and_then(|end| job.clock.get().map(|clock| clock.when(end)));
                    if heard_end.is_some_and(|at| at <= rune_core::time::now()) {
                        end_due = None;
                        drop(s);
                        (job.notify)();
                        s = pipe.lock();
                        continue;
                    }
                    if job.kind == Kind::Video && !playing_sound && touched.elapsed() > PARK {
                        s.news.push(News::Parked);
                        drop(s);
                        (job.notify)();
                        return Ok(());
                    }
                    let patience = if sound_full && playing_sound && !audio_wait {
                        SOUND_POLL
                    } else {
                        PARK
                    };
                    let patience = heard_end.map_or(patience, |at| {
                        patience.min(
                            at.saturating_duration_since(rune_core::time::now())
                                + Duration::from_millis(2),
                        )
                    });
                    s = pipe.wake.wait_timeout(s, patience).unwrap_or_else(|e| e.into_inner()).0;
                    waiting_since = waiting_since.max(touched);
                }
            }

            let decode_audio = self.output.is_some() || job.kind == Kind::Audio;
            match self.opened.next(decode_video, decode_audio)? {
                DecodedItem::Video { pts, frame } => {
                    if job.kind == Kind::Video {
                        ever_decoded = true;
                    }
                    let unwrapped = epoch + pts;
                    if pts + self.facts.meta.frame * 0.5 >= skip_before {
                        sent_video = true;
                        last_skipped = None;
                        let mut s = pipe.lock();
                        let was_empty = s.frames.is_empty();
                        s.frames.push_back(Decoded { generation, pts: unwrapped, frame });
                        drop(s);
                        if was_empty {
                            (job.notify)();
                        }
                    } else {
                        last_skipped = Some((unwrapped, frame));
                    }
                }
                DecodedItem::Audio { pts, samples } => {
                    if job.kind == Kind::Audio {
                        ever_decoded = true;
                    }
                    if let Some(o) = &self.output {
                        let spec = o.spec();
                        let frames = samples.len() / usize::from(spec.channels).max(1);
                        let chunk_end = epoch + pts + frames as f64 / f64::from(spec.rate);
                        if chunk_end > last_audio_end {
                            last_audio_end = chunk_end;
                        }
                        o.push(samples, epoch + pts, skip_before);
                    }
                }
                DecodedItem::Wait => {
                    audio_wait = true;
                }
                DecodedItem::Eof => {
                    let seeked_to_end = skip_before > 0.0
                        && self.facts.meta.duration > 0.0
                        && skip_before >= self.facts.meta.duration;
                    if !ever_decoded && !seeked_to_end {
                        return Err("no decodable frames in clip".into());
                    }
                    let looping = pipe.lock().looping;
                    if looping && self.facts.meta.duration > 0.0 {
                        epoch += self.facts.meta.duration;
                        skip_before = 0.0;
                        sent_video = false;
                        last_skipped = None;
                        audio_wait = false;
                        self.opened.seek(0.0)?;
                    } else {
                        let audio_end = if job.kind == Kind::Audio {
                            let dur_end = epoch + self.facts.meta.duration;
                            Some(if last_audio_end > epoch { last_audio_end } else { dur_end })
                        } else {
                            None
                        };
                        {
                            let mut s = pipe.lock();
                            if !sent_video
                                && let Some((pts, frame)) = last_skipped.take()
                            {
                                sent_video = true;
                                s.frames.push_back(Decoded { generation, pts, frame });
                            }
                            s.ended = Some(generation);
                            s.audio_end = audio_end;
                        }
                        if job.kind == Kind::Audio {
                            end_due = audio_end;
                        }
                        (job.notify)();
                    }
                }
            }
        }
    }
}
