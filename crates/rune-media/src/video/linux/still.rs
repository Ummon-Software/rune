//! Headless runs on Linux: a clip's facts and its picture at a virtual time, decoded
//! synchronously on the CPU, so `--snap` and `--frames` are deterministic. No player, no sound.

use super::super::{Kind, MediaSource, Meta};
use super::ffmpeg::{DecodedItem, Opened};
use crate::native_surface::ByteFrame;

pub(super) trait FrameSource: Send {
    fn frame(&mut self, at: f64) -> Result<ByteFrame, String>;
}

pub(crate) struct Still {
    opened: Opened,
    /// The last picture read at or before the time asked, and the one read after it: a later
    /// time reads on from them rather than seeking.
    current: Option<(f64, ByteFrame)>,
    next: Option<(f64, ByteFrame)>,
}

// SAFETY: `Still` is owned exclusively inside `Shared::native`'s mutex and never accessed concurrently.
#[allow(unsafe_code)]
unsafe impl Send for Still {}

/// Opens `source` and reads its facts; for video, keeps a decoder for its pictures too.
pub(super) fn open(
    source: &MediaSource,
    headers: &[(String, String)],
    kind: Kind,
) -> Result<(Option<Box<dyn FrameSource>>, Meta), String> {
    let opened = Opened::open(source, headers, kind, false)?;
    let meta = Meta {
        duration: opened.duration,
        size: opened.picture.map(|p| p.natural),
        frame: opened.picture.map_or(1.0 / 30.0, |p| p.frame),
        sound: opened.has_sound,
    };
    let still = opened.picture.map(|_| -> Box<dyn FrameSource> {
        Box::new(Still {
            opened,
            current: None,
            next: None,
        })
    });
    Ok((still, meta))
}

impl FrameSource for Still {
    /// The picture showing at clip second `at`.
    fn frame(&mut self, at: f64) -> Result<ByteFrame, String> {
        let shows = |t: f64| t <= at + 1e-6;
        let reads_on = self.current.as_ref().is_some_and(|(t, _)| shows(*t) && at - t < 1.0);
        if !reads_on {
            self.opened.seek(at)?;
            (self.current, self.next) = (None, None);
        }
        if let Some((t, _)) = &self.next {
            if !shows(*t) {
                return self
                    .current
                    .as_ref()
                    .map(|(_, f)| copy(f))
                    .ok_or_else(|| "no picture at that time".into());
            }
            self.current = self.next.take();
        }
        loop {
            match self.opened.next(true, false)? {
                DecodedItem::Video { pts, frame } => {
                    if !shows(pts) && self.current.is_some() {
                        self.next = Some((pts, frame));
                        break;
                    }
                    self.current = Some((pts, frame));
                }
                DecodedItem::Audio { .. } | DecodedItem::Wait => {}
                DecodedItem::Eof => break,
            }
        }
        self.current
            .as_ref()
            .map(|(_, f)| copy(f))
            .ok_or_else(|| "no picture at that time".into())
    }
}

fn copy(f: &ByteFrame) -> ByteFrame {
    ByteFrame {
        width: f.width,
        height: f.height,
        format: f.format,
        color: f.color,
        planes: f.planes.clone(),
    }
}
