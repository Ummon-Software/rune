//! FFmpeg demuxing, decoding, scaling and resampling for Linux.
//!
//! Dynamically links the system's shared libav* libraries through `ffmpeg-next`. Local files
//! open with `protocol_whitelist = file` so playlists or redirectors can never reach the network,
//! and extensionless files (such as content-addressed blob caches) probe cleanly by header bytes.

use super::super::{Kind, MediaSource};
use super::audio::OutputSpec;
use crate::native_surface::{ByteFrame, ColorInfo, Format, Matrix, Orient, Transfer};
use ffmpeg_next as ff;
use ff::codec::packet::side_data::Type as SideDataType;
use ff::color::{Range, Space};
use ff::format::stream::Disposition;
use ff::software::{resampling, scaling};
use ff::util::format::{Pixel, Sample, sample::Type as SampleType};
use ff::{ChannelLayout, Dictionary, Packet, Rational, frame, media};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static INIT: OnceLock<Result<(), String>> = OnceLock::new();

pub(super) fn init() -> Result<(), String> {
    INIT.get_or_init(|| {
        ff::init().map_err(|e| format!("FFmpeg couldn't start: {e}"))?;
        ff::log::set_level(ff::log::Level::Quiet);
        Ok(())
    })
    .clone()
}

/// The first video stream's picture geometry, color metadata and nominal cadence.
#[derive(Clone, Copy, Debug)]
pub(super) struct Picture {
    /// Display size in pixels (pixel aspect and rotation applied).
    pub natural: (u32, u32),
    pub format: Format,
    pub color: ColorInfo,
    pub orient: Orient,
    /// Seconds per frame, from the stream's nominal rate.
    pub frame: f64,
}

/// The first audio stream's native sample rate and channel count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sound {
    pub channels: u16,
    pub rate: u32,
}

pub(super) enum DecodedItem {
    Video { pts: f64, frame: ByteFrame },
    Audio { pts: f64, samples: Vec<f32> },
    Eof,
}

struct VideoTrack {
    index: usize,
    time_base: f64,
    start_time: f64,
    decoder: ff::decoder::Video,
    picture: Picture,
    scaler: Option<(Pixel, u32, u32, scaling::Context)>,
    frame: frame::Video,
    scaled: frame::Video,
    eof_sent: bool,
    eof: bool,
}

struct AudioTrack {
    index: usize,
    time_base: f64,
    start_time: f64,
    decoder: ff::decoder::Audio,
    resampler: Option<(Sample, ChannelLayout, u32, OutputSpec, resampling::Context)>,
    output_spec: Option<OutputSpec>,
    frame: frame::Audio,
    eof_sent: bool,
    eof: bool,
}

pub(super) struct Opened {
    ictx: ff::format::context::Input,
    video: Option<VideoTrack>,
    audio: Option<AudioTrack>,
    pub picture: Option<Picture>,
    pub sound: Option<Sound>,
    pub has_sound: bool,
    pub duration: f64,
    demux_eof: bool,
}

impl Opened {
    pub(super) fn open(
        source: &MediaSource,
        headers: &[(String, String)],
        kind: Kind,
        want_audio: bool,
    ) -> Result<Self, String> {
        init()?;
        let (target, opts) = resolve_source(source, headers)?;
        let ictx = ff::format::input_with_dictionary(&target, opts)
            .map_err(|e| format!("FFmpeg couldn't open the clip: {e}"))?;

        let container_dur = if ictx.duration() > 0 {
            ictx.duration() as f64 / f64::from(ff::ffi::AV_TIME_BASE)
        } else {
            0.0
        };

        let video_idx = if kind == Kind::Video {
            pick_video_stream(&ictx)
        } else {
            None
        };
        let audio_idx = ictx.streams().best(media::Type::Audio).map(|s| s.index());

        let (video, picture, video_dur) = match video_idx {
            Some(index) => {
                let (track, pic, dur) = open_video_track(&ictx, index)?;
                (Some(track), Some(pic), dur)
            }
            None if kind == Kind::Video => return Err("no playable video track".into()),
            None => (None, None, 0.0),
        };

        let mut has_sound = audio_idx.is_some();
        let (audio, sound, audio_dur) = match audio_idx.filter(|_| want_audio || kind == Kind::Audio) {
            Some(index) => match open_audio_track(&ictx, index) {
                Ok((track, snd, dur)) => (Some(track), Some(snd), dur),
                Err(e) if kind == Kind::Audio => return Err(e),
                Err(_) => {
                    has_sound = false;
                    (None, None, 0.0)
                }
            },
            None if kind == Kind::Audio => return Err("no playable audio track".into()),
            None => (None, None, 0.0),
        };

        let duration = if container_dur > 0.0 {
            container_dur
        } else {
            video_dur.max(audio_dur)
        };

        Ok(Self {
            ictx,
            video,
            audio,
            picture,
            sound,
            has_sound,
            duration,
            demux_eof: false,
        })
    }

    pub(super) fn set_audio_output(&mut self, spec: Option<OutputSpec>) {
        if let Some(audio) = &mut self.audio {
            audio.output_spec = spec;
            audio.resampler = None;
        }
    }

    pub(super) fn seek(&mut self, seconds: f64) -> Result<(), String> {
        let target = (seconds.max(0.0) * f64::from(ff::ffi::AV_TIME_BASE)) as i64;
        self.ictx
            .seek(target, ..target)
            .or_else(|_| self.ictx.seek(0, ..))
            .map_err(|e| format!("seek failed: {e}"))?;
        if let Some(v) = &mut self.video {
            v.decoder.flush();
            v.eof_sent = false;
            v.eof = false;
        }
        if let Some(a) = &mut self.audio {
            a.decoder.flush();
            a.eof_sent = false;
            a.eof = false;
        }
        self.demux_eof = false;
        Ok(())
    }

    /// Reads the next decoded video frame or resampled audio chunk, or `Eof` when all active
    /// tracks have drained.
    pub(super) fn next(&mut self, decode_audio: bool) -> Result<DecodedItem, String> {
        loop {
            if let Some(item) = self.receive_video()? {
                return Ok(item);
            }
            if decode_audio
                && let Some(item) = self.receive_audio()?
            {
                return Ok(item);
            }
            if self.demux_eof {
                let v_done = self.video.as_ref().is_none_or(|v| v.eof);
                let a_done = !decode_audio || self.audio.as_ref().is_none_or(|a| a.eof);
                if v_done && a_done {
                    return Ok(DecodedItem::Eof);
                }
                if let Some(v) = &mut self.video
                    && !v.eof_sent
                {
                    v.decoder.send_eof().ok();
                    v.eof_sent = true;
                }
                if decode_audio
                    && let Some(a) = &mut self.audio
                    && !a.eof_sent
                {
                    a.decoder.send_eof().ok();
                    a.eof_sent = true;
                }
                continue;
            }
            let mut packet = Packet::empty();
            match packet.read(&mut self.ictx) {
                Ok(()) => {
                    let idx = packet.stream();
                    if let Some(v) = &mut self.video
                        && v.index == idx
                    {
                        v.decoder.send_packet(&packet).ok();
                    } else if decode_audio
                        && let Some(a) = &mut self.audio
                        && a.index == idx
                    {
                        a.decoder.send_packet(&packet).ok();
                    }
                }
                Err(ff::Error::Eof) => {
                    self.demux_eof = true;
                }
                Err(e) => return Err(format!("decode read failed: {e}")),
            }
        }
    }

    fn receive_video(&mut self) -> Result<Option<DecodedItem>, String> {
        let Some(v) = &mut self.video else { return Ok(None) };
        if v.eof {
            return Ok(None);
        }
        match v.decoder.receive_frame(&mut v.frame) {
            Ok(()) => {
                let raw_pts = v.frame.timestamp().unwrap_or(0);
                let pts = (raw_pts as f64 * v.time_base - v.start_time).max(0.0);
                let bytes = video_frame_to_bytes(v)?;
                Ok(Some(DecodedItem::Video { pts, frame: bytes }))
            }
            Err(ff::Error::Other { errno }) if errno == libc::EAGAIN => Ok(None),
            Err(ff::Error::Eof) => {
                v.eof = true;
                Ok(None)
            }
            Err(e) => Err(format!("video decode error: {e}")),
        }
    }

    fn receive_audio(&mut self) -> Result<Option<DecodedItem>, String> {
        let Some(a) = &mut self.audio else { return Ok(None) };
        if a.eof {
            return Ok(None);
        }
        match a.decoder.receive_frame(&mut a.frame) {
            Ok(()) => {
                let raw_pts = a.frame.timestamp().unwrap_or(0);
                let pts = (raw_pts as f64 * a.time_base - a.start_time).max(0.0);
                let samples = audio_frame_to_f32(a)?;
                Ok(Some(DecodedItem::Audio { pts, samples }))
            }
            Err(ff::Error::Other { errno }) if errno == libc::EAGAIN => Ok(None),
            Err(ff::Error::Eof) => {
                a.eof = true;
                Ok(None)
            }
            Err(e) => Err(format!("audio decode error: {e}")),
        }
    }
}

fn pick_video_stream(ictx: &ff::format::context::Input) -> Option<usize> {
    if let Some(best) = ictx.streams().best(media::Type::Video)
        && !best.disposition().contains(Disposition::ATTACHED_PIC)
    {
        return Some(best.index());
    }
    ictx.streams()
        .find(|s| {
            s.parameters().medium() == media::Type::Video
                && !s.disposition().contains(Disposition::ATTACHED_PIC)
        })
        .map(|s| s.index())
}

fn open_video_track(
    ictx: &ff::format::context::Input,
    index: usize,
) -> Result<(VideoTrack, Picture, f64), String> {
    let stream = ictx.stream(index).ok_or("missing video stream")?;
    let time_base = rational_f64(stream.time_base()).unwrap_or(1.0 / 90_000.0);
    let start_time = if stream.start_time() != ff::ffi::AV_NOPTS_VALUE {
        stream.start_time() as f64 * time_base
    } else {
        0.0
    };
    let dur = if stream.duration() > 0 && stream.duration() != ff::ffi::AV_NOPTS_VALUE {
        stream.duration() as f64 * time_base
    } else {
        0.0
    };
    let rate = rational_f64(stream.avg_frame_rate())
        .or_else(|| rational_f64(stream.rate()))
        .filter(|r| *r > 0.5 && *r <= 240.0)
        .unwrap_or(30.0);
    let orient = stream_orient(&stream);

    let mut ctx = ff::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| format!("video codec context: {e}"))?;
    ctx.set_threading(ff::threading::Config {
        kind: ff::threading::Type::Frame,
        count: 0,
    });
    let decoder = ctx
        .decoder()
        .video()
        .map_err(|e| format!("no video decoder for this codec: {e}"))?;
    let (w, h) = (decoder.width(), decoder.height());
    if w == 0 || h == 0 {
        return Err("video track has no size".into());
    }
    let sar = rational_f64(decoder.aspect_ratio()).filter(|r| *r > 0.1 && *r < 10.0).unwrap_or(1.0);
    let disp_w = ((f64::from(w) * sar).round() as u32).max(1);
    let upright = orient.upright([disp_w, h]);
    let even = w.is_multiple_of(2) && h.is_multiple_of(2);
    let full = decoder.color_range() == Range::JPEG;
    let format = if !even {
        Format::Bgra
    } else if full {
        Format::Nv12Full
    } else {
        Format::Nv12Video
    };
    let matrix = match decoder.color_space() {
        Space::BT470BG | Space::SMPTE170M | Space::SMPTE240M => Matrix::Bt601,
        Space::BT709 => Matrix::Bt709,
        _ if h < 720 => Matrix::Bt601,
        _ => Matrix::Bt709,
    };
    let transfer = if format == Format::Bgra { Transfer::Srgb } else { Transfer::Bt709 };
    let picture = Picture {
        natural: (upright[0], upright[1]),
        format,
        color: ColorInfo { matrix, transfer },
        orient,
        frame: 1.0 / rate,
    };
    let track = VideoTrack {
        index,
        time_base,
        start_time,
        decoder,
        picture,
        scaler: None,
        frame: frame::Video::empty(),
        scaled: frame::Video::empty(),
        eof_sent: false,
        eof: false,
    };
    Ok((track, picture, dur))
}

fn open_audio_track(
    ictx: &ff::format::context::Input,
    index: usize,
) -> Result<(AudioTrack, Sound, f64), String> {
    let stream = ictx.stream(index).ok_or("missing audio stream")?;
    let time_base = rational_f64(stream.time_base()).unwrap_or(1.0 / 48_000.0);
    let start_time = if stream.start_time() != ff::ffi::AV_NOPTS_VALUE {
        stream.start_time() as f64 * time_base
    } else {
        0.0
    };
    let dur = if stream.duration() > 0 && stream.duration() != ff::ffi::AV_NOPTS_VALUE {
        stream.duration() as f64 * time_base
    } else {
        0.0
    };
    let ctx = ff::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| format!("audio codec context: {e}"))?;
    let decoder = ctx
        .decoder()
        .audio()
        .map_err(|e| format!("no audio decoder for this codec: {e}"))?;
    let sound = Sound {
        channels: decoder.channels().clamp(1, 8),
        rate: decoder.rate().max(8_000),
    };
    let track = AudioTrack {
        index,
        time_base,
        start_time,
        decoder,
        resampler: None,
        output_spec: None,
        frame: frame::Audio::empty(),
        eof_sent: false,
        eof: false,
    };
    Ok((track, sound, dur))
}

fn video_frame_to_bytes(v: &mut VideoTrack) -> Result<ByteFrame, String> {
    let w = v.frame.width();
    let h = v.frame.height();
    if w == 0 || h == 0 {
        return Err("decoded video frame has zero size".into());
    }
    let even = w.is_multiple_of(2) && h.is_multiple_of(2);
    let fmt = v.frame.format();
    let color = v.picture.color;
    let nv12_fmt = if v.frame.color_range() == Range::JPEG {
        Format::Nv12Full
    } else if v.frame.color_range() == Range::MPEG {
        Format::Nv12Video
    } else {
        match v.picture.format {
            Format::Nv12Full => Format::Nv12Full,
            _ => Format::Nv12Video,
        }
    };

    if even && fmt == Pixel::NV12 {
        return Ok(pack_nv12(&v.frame, w, h, nv12_fmt, color));
    }
    if even && fmt == Pixel::YUV420P {
        return Ok(pack_yuv420p_to_nv12(&v.frame, w, h, nv12_fmt, color));
    }

    let dst_pixel = if even { Pixel::NV12 } else { Pixel::BGRA };
    let needs_new = v
        .scaler
        .as_ref()
        .is_none_or(|(p, sw, sh, _)| *p != fmt || *sw != w || *sh != h);
    if needs_new {
        let scaler = scaling::Context::get(
            fmt,
            w,
            h,
            dst_pixel,
            w,
            h,
            scaling::Flags::BILINEAR,
        )
        .map_err(|e| format!("video format conversion failed: {e}"))?;
        v.scaler = Some((fmt, w, h, scaler));
    }
    let (_, _, _, scaler) = v.scaler.as_mut().ok_or("missing scaler")?;
    scaler
        .run(&v.frame, &mut v.scaled)
        .map_err(|e| format!("video scale failed: {e}"))?;
    if dst_pixel == Pixel::NV12 {
        Ok(pack_nv12(&v.scaled, w, h, nv12_fmt, color))
    } else {
        Ok(pack_bgra(&v.scaled, w, h, color))
    }
}

fn pack_nv12(frame: &frame::Video, w: u32, h: u32, format: Format, color: ColorInfo) -> ByteFrame {
    let (wu, hu) = (w as usize, h as usize);
    let mut y = Vec::with_capacity(wu * hu);
    let y_data = frame.data(0);
    let y_stride = frame.stride(0);
    for row in 0..hu {
        let start = row * y_stride;
        y.extend_from_slice(&y_data[start..start + wu]);
    }
    let mut uv = Vec::with_capacity(wu * hu / 2);
    let uv_data = frame.data(1);
    let uv_stride = frame.stride(1);
    for row in 0..hu / 2 {
        let start = row * uv_stride;
        uv.extend_from_slice(&uv_data[start..start + wu]);
    }
    ByteFrame { width: w, height: h, format, color, planes: [y, uv] }
}

fn pack_yuv420p_to_nv12(
    frame: &frame::Video,
    w: u32,
    h: u32,
    format: Format,
    color: ColorInfo,
) -> ByteFrame {
    let (wu, hu) = (w as usize, h as usize);
    let mut y = Vec::with_capacity(wu * hu);
    let y_data = frame.data(0);
    let y_stride = frame.stride(0);
    for row in 0..hu {
        let start = row * y_stride;
        y.extend_from_slice(&y_data[start..start + wu]);
    }
    let mut uv = Vec::with_capacity(wu * hu / 2);
    let u_data = frame.data(1);
    let u_stride = frame.stride(1);
    let v_data = frame.data(2);
    let v_stride = frame.stride(2);
    let cw = wu / 2;
    for row in 0..hu / 2 {
        let us = &u_data[row * u_stride..row * u_stride + cw];
        let vs = &v_data[row * v_stride..row * v_stride + cw];
        for (&u, &v) in us.iter().zip(vs) {
            uv.push(u);
            uv.push(v);
        }
    }
    ByteFrame { width: w, height: h, format, color, planes: [y, uv] }
}

fn pack_bgra(frame: &frame::Video, w: u32, h: u32, color: ColorInfo) -> ByteFrame {
    let (wu, hu) = (w as usize, h as usize);
    let mut bgra = Vec::with_capacity(wu * hu * 4);
    let data = frame.data(0);
    let stride = frame.stride(0);
    for row in 0..hu {
        let start = row * stride;
        bgra.extend_from_slice(&data[start..start + wu * 4]);
    }
    ByteFrame { width: w, height: h, format: Format::Bgra, color, planes: [bgra, Vec::new()] }
}

fn audio_frame_to_f32(a: &mut AudioTrack) -> Result<Vec<f32>, String> {
    let Some(spec) = a.output_spec else { return Ok(Vec::new()) };
    if a.frame.samples() == 0 {
        return Ok(Vec::new());
    }
    let src_fmt = a.frame.format();
    let src_rate = a.frame.rate().max(1);
    let src_layout = if a.frame.channel_layout().channels() > 0 {
        a.frame.channel_layout()
    } else {
        ChannelLayout::default(i32::from(a.frame.channels().max(1)))
    };
    let dst_layout = ChannelLayout::default(i32::from(spec.channels));
    let dst_fmt = Sample::F32(SampleType::Packed);

    let needs_new = a.resampler.as_ref().is_none_or(|(f, l, r, s, _)| {
        *f != src_fmt || *l != src_layout || *r != src_rate || *s != spec
    });
    if needs_new {
        let ctx = resampling::Context::get(
            src_fmt,
            src_layout,
            src_rate,
            dst_fmt,
            dst_layout,
            spec.rate,
        )
        .map_err(|e| format!("audio resampler setup failed: {e}"))?;
        a.resampler = Some((src_fmt, src_layout, src_rate, spec, ctx));
    }
    let (_, _, _, _, resampler) = a.resampler.as_mut().ok_or("missing resampler")?;
    let est = ((a.frame.samples() as u64 * u64::from(spec.rate)).div_ceil(u64::from(src_rate)) + 256)
        as usize;
    let mut out = frame::Audio::new(dst_fmt, est, dst_layout);
    let _ = resampler
        .run(&a.frame, &mut out)
        .map_err(|e| format!("audio resample failed: {e}"))?;
    let count = out.samples() * usize::from(spec.channels);
    let bytes = &out.data(0)[..count * 4];
    Ok(bytemuck::cast_slice::<u8, f32>(bytes).to_vec())
}

fn stream_orient(stream: &ff::format::stream::Stream<'_>) -> Orient {
    for sd in stream.side_data() {
        if sd.kind() == SideDataType::DisplayMatrix {
            let bytes = sd.data();
            if bytes.len() >= 36 {
                let m = |i: usize| {
                    let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap_or([0; 4]);
                    f64::from(i32::from_ne_bytes(b))
                };
                let (m0, m1, m3, m4) = (m(0), m(1), m(3), m(4));
                let s0 = m0.hypot(m3);
                let s1 = m1.hypot(m4);
                if s0 > 0.0 && s1 > 0.0 {
                    return Orient::from_transform(m0 / s0, m1 / s1, m3 / s0, m4 / s1);
                }
            }
        }
    }
    Orient::UP
}

fn rational_f64(r: Rational) -> Option<f64> {
    (r.denominator() > 0 && r.numerator() > 0)
        .then(|| f64::from(r.numerator()) / f64::from(r.denominator()))
}

fn resolve_source(
    source: &MediaSource,
    headers: &[(String, String)],
) -> Result<(String, Dictionary<'static>), String> {
    let mut opts = Dictionary::new();
    match source {
        MediaSource::File(path) => {
            let path = validate_local_file(path)?;
            opts.set("protocol_whitelist", "file");
            Ok((path, opts))
        }
        MediaSource::Url(url) if url.get(..5).is_some_and(|p| p.eq_ignore_ascii_case("file:")) => {
            let path = parse_file_url(url)?;
            let path = validate_local_file(&path)?;
            opts.set("protocol_whitelist", "file");
            Ok((path, opts))
        }
        MediaSource::Url(url) if url.starts_with("http://") || url.starts_with("https://") => {
            opts.set("protocol_whitelist", "http,https,tcp,tls,crypto");
            if !headers.is_empty() {
                let mut formatted = String::new();
                for (name, value) in headers {
                    if name.contains(['\r', '\n', ':']) || value.contains(['\r', '\n']) {
                        return Err("invalid HTTP request header".into());
                    }
                    formatted.push_str(name);
                    formatted.push_str(": ");
                    formatted.push_str(value);
                    formatted.push_str("\r\n");
                }
                opts.set("headers", &formatted);
            }
            Ok((url.clone(), opts))
        }
        MediaSource::Url(url) => Err(format!("unsupported URL: {url}")),
    }
}

fn validate_local_file(path: &Path) -> Result<String, String> {
    let raw = path
        .to_str()
        .ok_or_else(|| format!("unsupported local path: {}", path.display()))?;
    if raw.is_empty() || raw.chars().any(char::is_control) {
        return Err(format!("unsupported local path: {}", path.display()));
    }
    let meta = std::fs::metadata(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("no such file: {}", path.display()),
        _ => format!("can't open {}: {e}", path.display()),
    })?;
    if !meta.is_file() {
        return Err(format!("not a file: {}", path.display()));
    }
    Ok(raw.to_owned())
}

fn parse_file_url(url: &str) -> Result<PathBuf, String> {
    let rest = url
        .get(..7)
        .filter(|p| p.eq_ignore_ascii_case("file://"))
        .map(|_| &url[7..])
        .ok_or_else(|| format!("unsupported file URL: {url}"))?;
    let path = if let Some(s) = rest.strip_prefix('/') {
        format!("/{s}")
    } else if rest.get(..10).is_some_and(|p| p.eq_ignore_ascii_case("localhost/")) {
        format!("/{}", &rest[10..])
    } else {
        return Err(format!("remote file URL isn't supported: {url}"));
    };
    if path.contains(['?', '#']) {
        return Err(format!("unsupported file URL: {url}"));
    }
    let decoded = percent_decode(&path).ok_or_else(|| format!("invalid file URL encoding: {url}"))?;
    Ok(PathBuf::from(decoded))
}

fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = (*bytes.get(i + 1)? as char).to_digit(16)? as u8;
            let lo = (*bytes.get(i + 2)? as char).to_digit(16)? as u8;
            out.push((hi << 4) | lo);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}
