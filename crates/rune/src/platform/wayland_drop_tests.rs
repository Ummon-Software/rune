//! Unit tests for Wayland file drag-and-drop URI parsing and the drag state machine.

use std::path::PathBuf;

use super::{DragEffect, DragEvent, DragState, ReadOutcome, SurfaceKey, classify_uri_list, parse_uri_list, step};
use crate::window::FileStep;

const WIN_A: SurfaceKey = (10, 0x1000);
const WIN_B: SurfaceKey = (11, 0x2000);

fn run(state: &mut DragState, event: DragEvent) -> Vec<DragEffect> {
    step(state, event).into_iter().collect()
}

#[test]
fn uri_list_keeps_local_files_decoded() {
    let list = concat!(
        "# from a file manager\r\n",
        "file:///tmp/My%20Shot.png\r\n",
        "file://localhost/tmp/caf%C3%A9.mp4\r\n",
        "FILE://LOCALHOST/tmp/hash%231.txt?ignored=1#frag\r\n",
        "https://example.com/x\r\n",
        "file://other-host/c\r\n",
        "file:///tmp/bad%00nul.txt\r\n",
    );
    assert_eq!(
        parse_uri_list(list),
        [
            PathBuf::from("/tmp/My Shot.png"),
            PathBuf::from("/tmp/café.mp4"),
            PathBuf::from("/tmp/hash#1.txt"),
        ]
    );
}

#[test]
fn classify_uri_list_distinguishes_empty_and_non_file_payloads() {
    assert_eq!(classify_uri_list(""), ReadOutcome::Empty);
    assert_eq!(classify_uri_list("  \r\n "), ReadOutcome::Empty);
    assert_eq!(classify_uri_list("https://example.com/page\r\n"), ReadOutcome::NoLocalFiles);
    assert_eq!(
        classify_uri_list("file:///tmp/a.png\r\n"),
        ReadOutcome::Files(vec![PathBuf::from("/tmp/a.png")]),
    );
}

#[test]
fn normal_enter_read_motion_drop_flow() {
    let mut s = DragState::default();
    let files = vec![PathBuf::from("/tmp/a.png")];

    assert_eq!(
        run(&mut s, DragEvent::Enter {
            drag_id: 1,
            surface: Some(WIN_A),
            serial: 7,
            at: (12.0, 24.0),
            has_uri_list: true,
            has_offer: true,
        }),
        [DragEffect::Accept { serial: 7 }, DragEffect::StartRead { drag_id: 1 }],
    );

    // Motion before ReadDone updates the hover point without emitting Move before Enter.
    assert_eq!(run(&mut s, DragEvent::Motion { at: (30.0, 40.0) }), []);

    assert_eq!(
        run(&mut s, DragEvent::ReadDone { drag_id: 1, outcome: ReadOutcome::Files(files.clone()) }),
        [DragEffect::Step { surface: WIN_A, step: FileStep::Enter(files.clone(), Some((30.0, 40.0))) }],
    );

    assert_eq!(
        run(&mut s, DragEvent::Motion { at: (50.0, 60.0) }),
        [DragEffect::Step { surface: WIN_A, step: FileStep::Move((50.0, 60.0)) }],
    );

    assert_eq!(
        run(&mut s, DragEvent::Drop { read_id: 2 }),
        [
            DragEffect::Step { surface: WIN_A, step: FileStep::Drop(files, Some((50.0, 60.0))) },
            DragEffect::FinishAndDestroyOffer,
        ],
    );
    assert_eq!(s, DragState::Idle);

    // Trailing compositor Leave right after Drop is a no-op.
    assert_eq!(run(&mut s, DragEvent::Leave), []);
}

#[test]
fn fast_drop_survives_immediate_compositor_leave_before_read_done() {
    let mut s = DragState::default();
    let files = vec![PathBuf::from("/tmp/fast.png")];

    run(&mut s, DragEvent::Enter {
        drag_id: 1,
        surface: Some(WIN_A),
        serial: 3,
        at: (10.0, 20.0),
        has_uri_list: true,
        has_offer: true,
    });

    assert_eq!(
        run(&mut s, DragEvent::Drop { read_id: 2 }),
        [DragEffect::ArmDropTimeout { drag_id: 1 }],
    );

    // Per Wayland spec, Leave after Drop must NOT destroy the offer while transfer is pending.
    assert_eq!(run(&mut s, DragEvent::Leave), []);

    assert_eq!(
        run(&mut s, DragEvent::ReadDone { drag_id: 1, outcome: ReadOutcome::Files(files.clone()) }),
        [
            DragEffect::Step { surface: WIN_A, step: FileStep::Drop(files, Some((10.0, 20.0))) },
            DragEffect::FinishAndDestroyOffer,
        ],
    );
    assert_eq!(s, DragState::Idle);
}

#[test]
fn deferred_source_retries_receive_on_drop() {
    let mut s = DragState::default();
    let files = vec![PathBuf::from("/tmp/deferred.jpg")];

    run(&mut s, DragEvent::Enter {
        drag_id: 1,
        surface: Some(WIN_A),
        serial: 5,
        at: (5.0, 6.0),
        has_uri_list: true,
        has_offer: true,
    });
    assert_eq!(run(&mut s, DragEvent::ReadDone { drag_id: 1, outcome: ReadOutcome::Empty }), []);
    assert_eq!(run(&mut s, DragEvent::Motion { at: (15.0, 25.0) }), []);

    assert_eq!(
        run(&mut s, DragEvent::Drop { read_id: 2 }),
        [DragEffect::StartRead { drag_id: 2 }, DragEffect::ArmDropTimeout { drag_id: 2 }],
    );
    assert_eq!(run(&mut s, DragEvent::Leave), []);

    // Stale read id is ignored.
    assert_eq!(run(&mut s, DragEvent::ReadDone { drag_id: 1, outcome: ReadOutcome::Empty }), []);

    assert_eq!(
        run(&mut s, DragEvent::ReadDone { drag_id: 2, outcome: ReadOutcome::Files(files.clone()) }),
        [
            DragEffect::Step { surface: WIN_A, step: FileStep::Drop(files, Some((15.0, 25.0))) },
            DragEffect::FinishAndDestroyOffer,
        ],
    );
    assert_eq!(s, DragState::Idle);
}

#[test]
fn non_file_uri_list_rejects_and_leaves_cleanly() {
    let mut s = DragState::default();

    run(&mut s, DragEvent::Enter {
        drag_id: 1,
        surface: Some(WIN_A),
        serial: 9,
        at: (1.0, 2.0),
        has_uri_list: true,
        has_offer: true,
    });

    assert_eq!(
        run(&mut s, DragEvent::ReadDone { drag_id: 1, outcome: ReadOutcome::NoLocalFiles }),
        [DragEffect::Reject { serial: 9 }],
    );
    assert_eq!(s, DragState::Rejected);
    assert_eq!(run(&mut s, DragEvent::Motion { at: (9.0, 9.0) }), []);
    assert_eq!(run(&mut s, DragEvent::Leave), [DragEffect::DestroyOffer]);
    assert_eq!(s, DragState::Idle);
}

#[test]
fn foreign_surface_or_missing_mime_rejects_until_leave() {
    let mut s = DragState::default();

    assert_eq!(
        run(&mut s, DragEvent::Enter {
            drag_id: 1,
            surface: None,
            serial: 4,
            at: (0.0, 0.0),
            has_uri_list: true,
            has_offer: true,
        }),
        [DragEffect::Reject { serial: 4 }],
    );
    assert_eq!(run(&mut s, DragEvent::Leave), [DragEffect::DestroyOffer]);

    assert_eq!(
        run(&mut s, DragEvent::Enter {
            drag_id: 2,
            surface: Some(WIN_A),
            serial: 5,
            at: (0.0, 0.0),
            has_uri_list: false,
            has_offer: true,
        }),
        [DragEffect::Reject { serial: 5 }],
    );
    assert_eq!(run(&mut s, DragEvent::Leave), [DragEffect::DestroyOffer]);
}

#[test]
fn drop_timeout_and_window_close_clean_up_offers() {
    let mut s = DragState::default();
    run(&mut s, DragEvent::Enter {
        drag_id: 1,
        surface: Some(WIN_A),
        serial: 1,
        at: (1.0, 1.0),
        has_uri_list: true,
        has_offer: true,
    });
    run(&mut s, DragEvent::Drop { read_id: 2 });

    // Stale timeout ignored.
    assert_eq!(run(&mut s, DragEvent::DropTimeout { drag_id: 99 }), []);
    assert_eq!(run(&mut s, DragEvent::DropTimeout { drag_id: 1 }), [DragEffect::DestroyOffer]);
    assert_eq!(s, DragState::Idle);

    // Closing the active window destroys its offer without queueing steps.
    run(&mut s, DragEvent::Enter {
        drag_id: 3,
        surface: Some(WIN_A),
        serial: 2,
        at: (2.0, 2.0),
        has_uri_list: true,
        has_offer: true,
    });
    assert_eq!(run(&mut s, DragEvent::WindowClosed { surface: WIN_B }), []);
    assert_eq!(run(&mut s, DragEvent::WindowClosed { surface: WIN_A }), [DragEffect::DestroyOffer]);
    assert_eq!(s, DragState::Idle);
}

#[test]
fn reenter_on_second_window_leaves_first_and_destroys_previous_offer() {
    let mut s = DragState::default();
    let files = vec![PathBuf::from("/tmp/one.txt")];

    run(&mut s, DragEvent::Enter {
        drag_id: 1,
        surface: Some(WIN_A),
        serial: 1,
        at: (1.0, 1.0),
        has_uri_list: true,
        has_offer: true,
    });
    run(&mut s, DragEvent::ReadDone { drag_id: 1, outcome: ReadOutcome::Files(files) });

    assert_eq!(
        run(&mut s, DragEvent::Enter {
            drag_id: 2,
            surface: Some(WIN_B),
            serial: 2,
            at: (8.0, 9.0),
            has_uri_list: true,
            has_offer: true,
        }),
        [
            DragEffect::Step { surface: WIN_A, step: FileStep::Leave },
            DragEffect::DestroyPrevOffer,
            DragEffect::Accept { serial: 2 },
            DragEffect::StartRead { drag_id: 2 },
        ],
    );
}
