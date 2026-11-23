//! Pure state machine and `text/uri-list` parser for native Wayland file drag-and-drop.
//! Kept free of Wayland client bindings so its tests run headless on every platform.

use std::path::PathBuf;

use crate::machine::Effects;
use crate::window::FileStep;

/// Identifies a registered Rune window surface on a Wayland display (`wl_surface` protocol ID
/// and proxy pointer).
pub(crate) type SurfaceKey = (u32, usize);

/// What reading `text/uri-list` from a drag source yielded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReadOutcome {
    /// One or more local `file://` paths.
    Files(Vec<PathBuf>),
    /// Non-empty payload with no local `file://` URIs (for example a browser `https://` link),
    /// or an I/O error reading the pipe.
    NoLocalFiles,
    /// Empty payload (zero bytes or whitespace only): some sources ignore `receive` before
    /// `drop`, so the drag stays accepted and retries on `drop`.
    Empty,
}

/// One seat's file-drag state on a Wayland `wl_data_device`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) enum DragState {
    /// No active drag offer on this seat.
    #[default]
    Idle,
    /// An offer we refused (foreign surface, missing `text/uri-list`, or non-file URIs). Kept
    /// alive until `Leave` or the next `Enter` so it is destroyed per the Wayland protocol.
    Rejected,
    /// Drag entered a registered window with `text/uri-list`; we accepted and started reading
    /// the URI list on a reader thread (`drag_id`).
    Reading {
        drag_id: u64,
        surface: SurfaceKey,
        serial: u32,
        at: (f32, f32),
    },
    /// The enter-time read came back empty; keep the offer accepted and read again on `Drop`.
    Deferred {
        drag_id: u64,
        surface: SurfaceKey,
        at: (f32, f32),
    },
    /// Local files are known and the window has received `FileStep::Enter`.
    Hovering {
        drag_id: u64,
        surface: SurfaceKey,
        at: (f32, f32),
        files: Vec<PathBuf>,
    },
    /// `wl_data_device.drop` arrived before the URI list finished reading. Subsequent `Leave`
    /// events from the compositor are ignored until `ReadDone` or `DropTimeout`.
    Dropping {
        drag_id: u64,
        surface: SurfaceKey,
        at: (f32, f32),
    },
}

/// Inputs to one seat's Wayland drag state machine.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DragEvent {
    /// `wl_data_device.enter`. `surface` is `Some` when the entered surface is a registered
    /// Rune window.
    Enter {
        drag_id: u64,
        surface: Option<SurfaceKey>,
        serial: u32,
        at: (f32, f32),
        has_uri_list: bool,
        has_offer: bool,
    },
    /// `wl_data_device.motion`.
    Motion { at: (f32, f32) },
    /// `wl_data_device.leave`.
    Leave,
    /// `wl_data_device.drop`. `read_id` tags a fresh read when the state was `Deferred`.
    Drop { read_id: u64 },
    /// The reader thread finished reading `text/uri-list` for `drag_id`.
    ReadDone { drag_id: u64, outcome: ReadOutcome },
    /// A `Dropping` read timed out waiting for the source.
    DropTimeout { drag_id: u64 },
    /// A registered Rune window closed or hid while a drag may have targeted it.
    WindowClosed { surface: SurfaceKey },
}

/// Actions the Wayland adapter performs for a state transition, in order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DragEffect {
    /// Destroy the previous drag's offer replaced by a new `Enter`.
    DestroyPrevOffer,
    /// Tell the compositor this offer is not accepted (`accept(serial, None)` and v3
    /// `set_actions(None, None)`).
    Reject { serial: u32 },
    /// Tell the compositor we accept `text/uri-list` as a copy (`accept` + v3 `set_actions`).
    Accept { serial: u32 },
    /// Request `text/uri-list` via `wl_data_offer.receive` on a background reader tagged with
    /// `drag_id`.
    StartRead { drag_id: u64 },
    /// Queue a `FileStep` for `surface` and wake its event loop.
    Step { surface: SurfaceKey, step: FileStep },
    /// Finish a successful drop (`offer.finish()` when v3 with a negotiated action) and destroy
    /// the current offer.
    FinishAndDestroyOffer,
    /// Destroy the current offer without calling `finish()`.
    DestroyOffer,
    /// Start the drop-read timeout for `drag_id`.
    ArmDropTimeout { drag_id: u64 },
}

/// Advances one seat's drag state machine by `event`.
pub(crate) fn step(state: &mut DragState, event: DragEvent) -> Effects<DragEffect> {
    match event {
        DragEvent::Enter { drag_id, surface, serial, at, has_uri_list, has_offer } => {
            let mut effects = Effects::none();
            match std::mem::take(state) {
                DragState::Idle => {}
                DragState::Hovering { surface: old, .. } => {
                    effects.push(DragEffect::Step { surface: old, step: FileStep::Leave });
                    effects.push(DragEffect::DestroyPrevOffer);
                }
                DragState::Rejected
                | DragState::Reading { .. }
                | DragState::Deferred { .. }
                | DragState::Dropping { .. } => {
                    effects.push(DragEffect::DestroyPrevOffer);
                }
            }
            if !has_offer {
                return effects;
            }
            match (surface, has_uri_list) {
                (Some(surface), true) => {
                    *state = DragState::Reading { drag_id, surface, serial, at };
                    effects.push(DragEffect::Accept { serial });
                    effects.push(DragEffect::StartRead { drag_id });
                }
                _ => {
                    *state = DragState::Rejected;
                    effects.push(DragEffect::Reject { serial });
                }
            }
            effects
        }
        DragEvent::Motion { at } => match state {
            DragState::Reading { at: cur, .. } | DragState::Deferred { at: cur, .. } => {
                *cur = at;
                Effects::none()
            }
            DragState::Hovering { surface, at: cur, .. } => {
                *cur = at;
                Effects::one(DragEffect::Step { surface: *surface, step: FileStep::Move(at) })
            }
            DragState::Idle | DragState::Rejected | DragState::Dropping { .. } => Effects::none(),
        },
        DragEvent::Leave => match state {
            // Compositors (including wlroots/Hyprland) commonly send `leave` right after `drop`;
            // keep the offer alive until the drop transfer finishes or times out.
            DragState::Idle | DragState::Dropping { .. } => Effects::none(),
            DragState::Hovering { surface, .. } => {
                let surface = *surface;
                *state = DragState::Idle;
                Effects::two(
                    DragEffect::Step { surface, step: FileStep::Leave },
                    DragEffect::DestroyOffer,
                )
            }
            DragState::Rejected | DragState::Reading { .. } | DragState::Deferred { .. } => {
                *state = DragState::Idle;
                Effects::one(DragEffect::DestroyOffer)
            }
        },
        DragEvent::Drop { read_id } => match std::mem::take(state) {
            DragState::Idle => Effects::none(),
            DragState::Rejected => Effects::one(DragEffect::DestroyOffer),
            DragState::Hovering { surface, at, files, .. } => Effects::two(
                DragEffect::Step { surface, step: FileStep::Drop(files, Some(at)) },
                DragEffect::FinishAndDestroyOffer,
            ),
            DragState::Reading { drag_id, surface, at, .. } => {
                *state = DragState::Dropping { drag_id, surface, at };
                Effects::one(DragEffect::ArmDropTimeout { drag_id })
            }
            DragState::Deferred { surface, at, .. } => {
                *state = DragState::Dropping { drag_id: read_id, surface, at };
                Effects::two(
                    DragEffect::StartRead { drag_id: read_id },
                    DragEffect::ArmDropTimeout { drag_id: read_id },
                )
            }
            DragState::Dropping { drag_id, surface, at } => {
                *state = DragState::Dropping { drag_id, surface, at };
                Effects::none()
            }
        },
        DragEvent::ReadDone { drag_id, outcome } => match state {
            DragState::Reading { drag_id: cur, surface, serial, at } if *cur == drag_id => {
                let (surface, serial, at) = (*surface, *serial, *at);
                match outcome {
                    ReadOutcome::Files(files) if !files.is_empty() => {
                        *state = DragState::Hovering {
                            drag_id,
                            surface,
                            at,
                            files: files.clone(),
                        };
                        Effects::one(DragEffect::Step {
                            surface,
                            step: FileStep::Enter(files, Some(at)),
                        })
                    }
                    ReadOutcome::Empty => {
                        *state = DragState::Deferred { drag_id, surface, at };
                        Effects::none()
                    }
                    ReadOutcome::NoLocalFiles | ReadOutcome::Files(_) => {
                        *state = DragState::Rejected;
                        Effects::one(DragEffect::Reject { serial })
                    }
                }
            }
            DragState::Dropping { drag_id: cur, surface, at } if *cur == drag_id => {
                let (surface, at) = (*surface, *at);
                *state = DragState::Idle;
                match outcome {
                    ReadOutcome::Files(files) if !files.is_empty() => Effects::two(
                        DragEffect::Step { surface, step: FileStep::Drop(files, Some(at)) },
                        DragEffect::FinishAndDestroyOffer,
                    ),
                    ReadOutcome::Empty | ReadOutcome::NoLocalFiles | ReadOutcome::Files(_) => {
                        Effects::one(DragEffect::DestroyOffer)
                    }
                }
            }
            _ => Effects::none(),
        },
        DragEvent::DropTimeout { drag_id } => match state {
            DragState::Dropping { drag_id: cur, .. } if *cur == drag_id => {
                *state = DragState::Idle;
                Effects::one(DragEffect::DestroyOffer)
            }
            _ => Effects::none(),
        },
        DragEvent::WindowClosed { surface: closed } => match state {
            DragState::Reading { surface, .. }
            | DragState::Deferred { surface, .. }
            | DragState::Hovering { surface, .. }
            | DragState::Dropping { surface, .. }
                if *surface == closed =>
            {
                *state = DragState::Idle;
                Effects::one(DragEffect::DestroyOffer)
            }
            _ => Effects::none(),
        },
    }
}

/// Classifies the raw bytes read from a `text/uri-list` offer.
pub(crate) fn classify_uri_list(raw: &str) -> ReadOutcome {
    if raw.trim().is_empty() {
        return ReadOutcome::Empty;
    }
    let files = parse_uri_list(raw);
    if files.is_empty() {
        ReadOutcome::NoLocalFiles
    } else {
        ReadOutcome::Files(files)
    }
}

/// Parses local file paths from an RFC 2483 `text/uri-list`: `file://` URIs with empty or
/// `localhost` authority, percent-decoded; comments, non-file schemes, remote hosts and paths
/// containing NUL bytes are skipped.
pub(crate) fn parse_uri_list(list: &str) -> Vec<PathBuf> {
    list.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(strip_file_scheme)
        .filter_map(local_file_path)
        .filter_map(percent_decode_path)
        .collect()
}

fn strip_file_scheme(uri: &str) -> Option<&str> {
    let prefix = uri.get(..7)?;
    prefix.eq_ignore_ascii_case("file://").then(|| &uri[7..])
}

/// Extracts the `/...` path component for empty (`file:///path`) or `localhost`
/// (`file://localhost/path`) authority, stripping any raw `?query` or `#fragment`.
fn local_file_path(after_scheme: &str) -> Option<&str> {
    let path = if after_scheme.starts_with('/') {
        after_scheme
    } else if after_scheme.get(..9).is_some_and(|h| h.eq_ignore_ascii_case("localhost"))
        && after_scheme.as_bytes().get(9) == Some(&b'/')
    {
        &after_scheme[9..]
    } else {
        return None;
    };
    let path = path.split(['?', '#']).next().unwrap_or(path);
    (!path.is_empty()).then_some(path)
}

fn percent_decode_path(s: &str) -> Option<PathBuf> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%'
            && let (Some(h), Some(l)) = (
                b.get(i + 1).and_then(|c| hex(*c)),
                b.get(i + 2).and_then(|c| hex(*c)),
            )
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    if out.contains(&0) {
        return None;
    }
    #[cfg(unix)]
    {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        Some(PathBuf::from(OsString::from_vec(out)))
    }
    #[cfg(not(unix))]
    {
        Some(PathBuf::from(String::from_utf8_lossy(&out).into_owned()))
    }
}

#[cfg(test)]
#[path = "wayland_drop_tests.rs"]
mod tests;
