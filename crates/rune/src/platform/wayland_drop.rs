//! Files dragged in from other apps on Wayland, where winit 0.30 binds no `wl_data_device`.
//!
//! Each Wayland display connection gets one shared worker thread (`rune-wayland-drop`) owning a
//! private `EventQueue` on winit's `wl_display` (`Backend::from_foreign_display`) with a
//! `wl_data_device` per seat. The worker participates in libwayland's multi-threaded
//! `prepare_read` / `poll` / `read` protocol alongside winit's `WaylandSource`, so drag enter,
//! motion, leave and drop events are dispatched immediately even while winit's own queue is idle.
//! Reading `text/uri-list` happens on a short-lived pipe-reader thread so the compositor socket
//! never stalls; whenever a `FileStep` is produced, the worker wakes the target window's winit
//! event loop via `WakeEvent::FileDrop`, and `Host::flush_files` drains the steps on that turn.
//!
//! Rune's clipboard (`arboard` with `wayland-data-control`) uses its own connection and data
//! control protocol; any clipboard `selection` offer announced on this `wl_data_device` is
//! destroyed immediately so it neither leaks nor interferes.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::io::{PipeReader, PipeWriter, Read, Write};
use std::os::fd::{AsFd, AsRawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use wayland_backend::client::{Backend, ObjectId, WaylandError};
use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::wl_data_device::{self, WlDataDevice};
use wayland_client::protocol::wl_data_device_manager::{DndAction, WlDataDeviceManager};
use wayland_client::protocol::wl_data_offer::{self, WlDataOffer};
use wayland_client::protocol::wl_registry::{self, WlRegistry};
use wayland_client::protocol::wl_seat::{self, WlSeat};
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum, event_created_child};
use winit::window::Window;

use super::wayland_drop_machine::{
    DragEffect, DragEvent, DragState, ReadOutcome, SurfaceKey, classify_uri_list, step,
};
use crate::runtime::Waker;
use crate::time::Instant;
use crate::window::FileStep;

const URI_LIST: &str = "text/uri-list";
/// How long a drop waits for a drag source to finish writing `text/uri-list`.
const DROP_READ: Duration = Duration::from_millis(1000);
/// Cap on bytes read from a single `text/uri-list` transfer (1 MiB).
const MAX_URI_LIST_BYTES: u64 = 1024 * 1024;

/// Per-window Wayland drop target held inside `Win::native` so it unregisters and joins its
/// worker before the window and `wl_display` are destroyed.
#[derive(Default)]
pub(crate) enum Slot {
    #[default]
    Uninit,
    Unavailable,
    Active(WindowTarget),
}

impl Slot {
    /// Registers the window on first use and drains any `FileStep`s queued since the last call.
    pub(crate) fn take(
        &mut self,
        window: &Window,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Vec<FileStep> {
        if matches!(self, Slot::Uninit) {
            *self = match WindowTarget::new(window, Arc::new(wake)) {
                Some(target) => Slot::Active(target),
                None => Slot::Unavailable,
            };
        }
        match self {
            Slot::Active(target) => target.take(),
            Slot::Uninit | Slot::Unavailable => Vec::new(),
        }
    }

    /// Drains pending steps without initializing a new target (used while a window is hidden).
    pub(crate) fn drain_if_active(&mut self) -> Vec<FileStep> {
        match self {
            Slot::Active(target) => target.take(),
            Slot::Uninit | Slot::Unavailable => Vec::new(),
        }
    }
}

/// Active registration for one Wayland window surface.
pub(crate) struct WindowTarget {
    worker: Arc<DisplayWorker>,
    surface: SurfaceKey,
    steps: Arc<WindowSteps>,
}

impl WindowTarget {
    fn new(window: &Window, wake: Waker) -> Option<Self> {
        let (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(handle)) = (
            window.display_handle().ok()?.as_raw(),
            window.window_handle().ok()?.as_raw(),
        ) else {
            return None;
        };
        // SAFETY: `handle.surface` is winit's live `wl_surface` proxy for `window`; we only
        // read its class/protocol ID here to build a `SurfaceKey`.
        let id = unsafe {
            ObjectId::from_ptr(WlSurface::interface(), handle.surface.as_ptr().cast())
        }
        .ok()?;
        let surface = (id.protocol_id(), handle.surface.as_ptr() as usize);
        let worker = DisplayWorker::get_or_start(display.display.as_ptr())?;
        let steps = Arc::new(WindowSteps::new(wake));
        worker.register(surface, Arc::clone(&steps));
        Some(WindowTarget { worker, surface, steps })
    }

    fn take(&self) -> Vec<FileStep> {
        self.steps.take()
    }
}

impl Drop for WindowTarget {
    fn drop(&mut self) {
        self.worker.unregister(self.surface);
    }
}

/// Queued steps for one window, with coalesced wake notifications.
struct WindowSteps {
    steps: Mutex<Vec<FileStep>>,
    wake_pending: AtomicBool,
    wake: Waker,
}

impl WindowSteps {
    fn new(wake: Waker) -> Self {
        Self { steps: Mutex::new(Vec::new()), wake_pending: AtomicBool::new(false), wake }
    }

    fn push(&self, step: FileStep) {
        let should_wake = {
            let mut steps = self.steps.lock().unwrap_or_else(|e| e.into_inner());
            if matches!(step, FileStep::Move(_))
                && matches!(steps.last(), Some(FileStep::Move(_)))
            {
                steps.pop();
            }
            steps.push(step);
            !self.wake_pending.swap(true, Ordering::AcqRel)
        };
        if should_wake {
            (self.wake)();
        }
    }

    fn take(&self) -> Vec<FileStep> {
        let mut steps = self.steps.lock().unwrap_or_else(|e| e.into_inner());
        self.wake_pending.store(false, Ordering::Release);
        std::mem::take(&mut *steps)
    }
}

/// Non-blocking write end of the worker's wakeup pipe.
struct WakePipe(Mutex<PipeWriter>);

impl WakePipe {
    fn poke(&self) {
        if let Ok(mut w) = self.0.lock() {
            w.write_all(&[1u8]).ok();
        }
    }
}

enum Cmd {
    ReadDone { seat_name: u32, drag_id: u64, outcome: ReadOutcome },
    WindowClosed { surface: SurfaceKey },
    Shutdown,
}

static WORKERS: Mutex<Option<HashMap<usize, Weak<DisplayWorker>>>> = Mutex::new(None);

/// One background Wayland data-device worker per `wl_display`, shared by all open windows on
/// that display and torn down cleanly when the last `WindowTarget` drops.
struct DisplayWorker {
    display_key: usize,
    windows: Arc<Mutex<HashMap<SurfaceKey, Arc<WindowSteps>>>>,
    cmd_tx: Sender<Cmd>,
    wake_pipe: Arc<WakePipe>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl DisplayWorker {
    fn get_or_start(display_ptr: *mut c_void) -> Option<Arc<Self>> {
        let key = display_ptr as usize;
        let mut map = WORKERS.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        if let Some(existing) = map.get(&key).and_then(Weak::upgrade) {
            return Some(existing);
        }
        let worker = Arc::new(Self::start(display_ptr)?);
        map.insert(key, Arc::downgrade(&worker));
        Some(worker)
    }

    fn start(display_ptr: *mut c_void) -> Option<Self> {
        // SAFETY: winit owns this `wl_display` for the life of the event loop, and every
        // `WindowTarget` drops (joining this worker thread) before its `Window` and `EventLoop`
        // are destroyed; a foreign backend never closes the display.
        let backend = unsafe { Backend::from_foreign_display(display_ptr.cast()) };
        let conn = Connection::from_backend(backend);
        let (globals, queue) = registry_queue_init::<WorkerState>(&conn).ok()?;
        let qh = queue.handle();
        let manager = globals.bind::<WlDataDeviceManager, _, _>(&qh, 1..=3, ()).ok()?;
        let registry = globals.registry().clone();

        let (wake_read, wake_write) = std::io::pipe().ok()?;
        set_nonblocking(wake_read.as_raw_fd()).ok()?;
        set_nonblocking(wake_write.as_raw_fd()).ok()?;

        let (cmd_tx, cmd_rx) = mpsc::channel();
        let wake_pipe = Arc::new(WakePipe(Mutex::new(wake_write)));
        let windows = Arc::new(Mutex::new(HashMap::new()));

        let mut state = WorkerState {
            conn: conn.clone(),
            registry,
            manager,
            windows: Arc::clone(&windows),
            cmd_tx: cmd_tx.clone(),
            wake_pipe: Arc::clone(&wake_pipe),
            devices: HashMap::new(),
            next_drag_id: 1,
            shutdown: false,
        };

        let seat_globals: Vec<(u32, u32)> = globals.contents().with_list(|list| {
            list.iter()
                .filter(|g| g.interface == "wl_seat")
                .map(|g| (g.name, g.version.min(5)))
                .collect()
        });
        for (name, version) in seat_globals {
            state.bind_seat(name, version, &qh);
        }
        conn.flush().ok()?;

        let thread = std::thread::Builder::new()
            .name("rune-wayland-drop".into())
            .spawn(move || run_worker(conn, queue, state, cmd_rx, wake_read))
            .ok()?;

        Some(DisplayWorker {
            display_key: display_ptr as usize,
            windows,
            cmd_tx,
            wake_pipe,
            thread: Mutex::new(Some(thread)),
        })
    }

    fn register(&self, surface: SurfaceKey, steps: Arc<WindowSteps>) {
        self.windows.lock().unwrap_or_else(|e| e.into_inner()).insert(surface, steps);
    }

    fn unregister(&self, surface: SurfaceKey) {
        self.windows.lock().unwrap_or_else(|e| e.into_inner()).remove(&surface);
        self.cmd_tx.send(Cmd::WindowClosed { surface }).ok();
        self.wake_pipe.poke();
    }
}

impl Drop for DisplayWorker {
    fn drop(&mut self) {
        self.cmd_tx.send(Cmd::Shutdown).ok();
        self.wake_pipe.poke();
        if let Some(handle) = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take() {
            handle.join().ok();
        }
        if let Ok(mut guard) = WORKERS.lock()
            && let Some(map) = guard.as_mut()
            && map.get(&self.display_key).is_some_and(|w| w.strong_count() == 0)
        {
            map.remove(&self.display_key);
        }
    }
}

fn set_nonblocking(fd: std::os::fd::RawFd) -> std::io::Result<()> {
    // SAFETY: `fcntl(F_GETFL)` / `F_SETFL` on a live pipe fd owned by this process.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

fn drain_pipe(reader: &mut PipeReader) {
    let mut buf = [0u8; 64];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

fn run_worker(
    conn: Connection,
    mut queue: EventQueue<WorkerState>,
    mut state: WorkerState,
    cmd_rx: Receiver<Cmd>,
    mut wake_read: PipeReader,
) {
    let wayland_fd = conn.backend().poll_fd().as_raw_fd();
    let wake_fd = wake_read.as_raw_fd();

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            state.handle_cmd(cmd);
        }
        if state.shutdown {
            break;
        }
        state.check_timeouts(crate::time::now());

        if queue.dispatch_pending(&mut state).is_err() {
            break;
        }
        if let Err(WaylandError::Io(e)) = conn.flush()
            && e.kind() != std::io::ErrorKind::WouldBlock
        {
            break;
        }

        let Some(guard) = queue.prepare_read() else {
            continue;
        };

        let timeout_ms = state.next_timeout_ms(crate::time::now());
        let mut fds = [
            libc::pollfd { fd: wayland_fd, events: libc::POLLIN, revents: 0 },
            libc::pollfd { fd: wake_fd, events: libc::POLLIN, revents: 0 },
        ];
        // SAFETY: `fds` points to two live descriptors (`wl_display` outlives this worker via
        // `DisplayWorker::drop` join, and `wake_read` is owned by this function).
        let rc = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, timeout_ms) };
        if rc < 0 {
            drop(guard);
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }

        let wl_revents = fds[0].revents;
        if wl_revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            drop(guard);
            break;
        }
        if wl_revents & libc::POLLIN != 0 {
            if let Err(WaylandError::Io(e)) = guard.read()
                && e.kind() != std::io::ErrorKind::WouldBlock
            {
                break;
            }
        } else {
            drop(guard);
        }

        if fds[1].revents & libc::POLLIN != 0 {
            drain_pipe(&mut wake_read);
        }
    }

    state.cleanup();
    conn.flush().ok();
}

/// MIME types and negotiated v3 DnD action announced on one `wl_data_offer`.
struct OfferData {
    mimes: Mutex<Vec<String>>,
    action: Mutex<DndAction>,
}

impl Default for OfferData {
    fn default() -> Self {
        Self { mimes: Mutex::new(Vec::new()), action: Mutex::new(DndAction::None) }
    }
}

impl OfferData {
    fn has_uri_list(&self) -> bool {
        self.mimes.lock().is_ok_and(|m| m.iter().any(|t| t == URI_LIST))
    }

    fn action(&self) -> DndAction {
        self.action.lock().map_or(DndAction::None, |a| *a)
    }
}

struct SeatDevice {
    seat: WlSeat,
    device: WlDataDevice,
    drag: DragState,
    offer: Option<WlDataOffer>,
    pending_offers: Vec<WlDataOffer>,
    drop_deadline: Option<(u64, Instant)>,
}

impl SeatDevice {
    fn destroy_all(mut self) {
        if let Some(offer) = self.offer.take() {
            offer.destroy();
        }
        for offer in self.pending_offers.drain(..) {
            offer.destroy();
        }
        if self.device.version() >= 2 {
            self.device.release();
        }
        if self.seat.version() >= 5 {
            self.seat.release();
        }
    }
}

struct WorkerState {
    conn: Connection,
    registry: WlRegistry,
    manager: WlDataDeviceManager,
    windows: Arc<Mutex<HashMap<SurfaceKey, Arc<WindowSteps>>>>,
    cmd_tx: Sender<Cmd>,
    wake_pipe: Arc<WakePipe>,
    devices: HashMap<u32, SeatDevice>,
    next_drag_id: u64,
    shutdown: bool,
}

impl WorkerState {
    fn alloc_drag_id(&mut self) -> u64 {
        let id = self.next_drag_id;
        self.next_drag_id = self.next_drag_id.wrapping_add(1).max(1);
        id
    }

    fn bind_seat(&mut self, name: u32, version: u32, qh: &QueueHandle<Self>) {
        if self.devices.contains_key(&name) {
            return;
        }
        let seat = self.registry.bind::<WlSeat, _, _>(name, version.min(5), qh, ());
        let device = self.manager.get_data_device(&seat, qh, name);
        self.devices.insert(
            name,
            SeatDevice {
                seat,
                device,
                drag: DragState::Idle,
                offer: None,
                pending_offers: Vec::new(),
                drop_deadline: None,
            },
        );
    }

    fn handle_cmd(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Shutdown => self.shutdown = true,
            Cmd::ReadDone { seat_name, drag_id, outcome } => {
                self.step_seat(seat_name, DragEvent::ReadDone { drag_id, outcome }, None);
            }
            Cmd::WindowClosed { surface } => {
                let seat_names: Vec<u32> = self.devices.keys().copied().collect();
                for seat_name in seat_names {
                    self.step_seat(seat_name, DragEvent::WindowClosed { surface }, None);
                }
            }
        }
    }

    fn check_timeouts(&mut self, now: Instant) {
        let expired: Vec<(u32, u64)> = self
            .devices
            .iter()
            .filter_map(|(&seat, dev)| {
                dev.drop_deadline.filter(|&(_, due)| due <= now).map(|(id, _)| (seat, id))
            })
            .collect();
        for (seat_name, drag_id) in expired {
            self.step_seat(seat_name, DragEvent::DropTimeout { drag_id }, None);
        }
    }

    fn next_timeout_ms(&self, now: Instant) -> libc::c_int {
        let Some(earliest) = self.devices.values().filter_map(|d| d.drop_deadline.map(|(_, t)| t)).min() else {
            return -1;
        };
        let ms = earliest.saturating_duration_since(now).as_millis();
        ms.min(libc::c_int::MAX as u128) as libc::c_int
    }

    fn step_seat(&mut self, seat_name: u32, event: DragEvent, mut prev_offer: Option<WlDataOffer>) {
        let Some(dev) = self.devices.get_mut(&seat_name) else {
            if let Some(old) = prev_offer { old.destroy(); }
            return;
        };
        for effect in step(&mut dev.drag, event) {
            match effect {
                DragEffect::DestroyPrevOffer => {
                    if let Some(old) = prev_offer.take() { old.destroy(); }
                    dev.drop_deadline = None;
                }
                DragEffect::Reject { serial } => if let Some(offer) = dev.offer.as_ref() {
                    offer.accept(serial, None);
                    if offer.version() >= 3 { offer.set_actions(DndAction::None, DndAction::None); }
                },
                DragEffect::Accept { serial } => if let Some(offer) = dev.offer.as_ref() {
                    offer.accept(serial, Some(URI_LIST.to_owned()));
                    if offer.version() >= 3 { offer.set_actions(DndAction::Copy, DndAction::Copy); }
                },
                DragEffect::StartRead { drag_id } => if let Some(offer) = dev.offer.as_ref() {
                    spawn_uri_reader(&self.conn, offer, seat_name, drag_id, self.cmd_tx.clone(), Arc::clone(&self.wake_pipe));
                },
                DragEffect::Step { surface, step } => {
                    if let Some(win) = self.windows.lock().unwrap_or_else(|e| e.into_inner()).get(&surface).cloned() {
                        win.push(step);
                    }
                }
                DragEffect::FinishAndDestroyOffer => {
                    if let Some(offer) = dev.offer.take() {
                        let action = offer.data::<OfferData>().map_or(DndAction::None, OfferData::action);
                        if offer.version() >= 3 && action != DndAction::None { offer.finish(); }
                        offer.destroy();
                    }
                    dev.drop_deadline = None;
                }
                DragEffect::DestroyOffer => {
                    if let Some(offer) = dev.offer.take() { offer.destroy(); }
                    dev.drop_deadline = None;
                }
                DragEffect::ArmDropTimeout { drag_id } => {
                    dev.drop_deadline = Some((drag_id, crate::time::now() + DROP_READ));
                }
            }
        }
        if let Some(old) = prev_offer { old.destroy(); }
    }

    fn cleanup(&mut self) {
        for (_, dev) in self.devices.drain() { dev.destroy_all(); }
    }
}

fn spawn_uri_reader(conn: &Connection, offer: &WlDataOffer, seat_name: u32, drag_id: u64, cmd_tx: Sender<Cmd>, wake_pipe: Arc<WakePipe>) {
    let fail = |tx: &Sender<Cmd>, wake: &WakePipe| {
        tx.send(Cmd::ReadDone { seat_name, drag_id, outcome: ReadOutcome::NoLocalFiles }).ok();
        wake.poke();
    };
    let Ok((reader, writer)) = std::io::pipe() else { fail(&cmd_tx, &wake_pipe); return; };
    offer.receive(URI_LIST.to_owned(), writer.as_fd());
    // Flush the `receive` request carrying `writer` before dropping our write end so the drag
    // source holds the only write descriptor and our reader reaches EOF once the source closes it.
    let flushed = conn.flush().is_ok();
    drop(writer);
    if !flushed { fail(&cmd_tx, &wake_pipe); return; }
    let (tx, wake) = (cmd_tx.clone(), Arc::clone(&wake_pipe));
    let spawned = std::thread::Builder::new().name("rune-wayland-drop-read".into()).spawn(move || {
        let mut bytes = Vec::new();
        let outcome = match reader.take(MAX_URI_LIST_BYTES + 1).read_to_end(&mut bytes) {
            Ok(_) => {
                if bytes.len() > MAX_URI_LIST_BYTES as usize { bytes.truncate(MAX_URI_LIST_BYTES as usize); }
                classify_uri_list(&String::from_utf8_lossy(&bytes))
            }
            Err(_) => ReadOutcome::NoLocalFiles,
        };
        tx.send(Cmd::ReadDone { seat_name, drag_id, outcome }).ok();
        wake.poke();
    });
    if spawned.is_err() { fail(&cmd_tx, &wake_pipe); }
}

impl Dispatch<WlRegistry, GlobalListContents> for WorkerState {
    fn event(state: &mut Self, _: &WlRegistry, event: wl_registry::Event, _: &GlobalListContents, _: &Connection, qh: &QueueHandle<Self>) {
        match event {
            wl_registry::Event::Global { name, interface, version } if interface == "wl_seat" => state.bind_seat(name, version, qh),
            wl_registry::Event::GlobalRemove { name } => if let Some(dev) = state.devices.remove(&name) {
                if let DragState::Hovering { surface, .. } = dev.drag
                    && let Some(win) = state.windows.lock().unwrap_or_else(|e| e.into_inner()).get(&surface).cloned()
                {
                    win.push(FileStep::Leave);
                }
                dev.destroy_all();
            },
            _ => {}
        }
    }
}

impl Dispatch<WlSeat, ()> for WorkerState {
    fn event(_: &mut Self, _: &WlSeat, _: wl_seat::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlDataDeviceManager, ()> for WorkerState {
    fn event(_: &mut Self, _: &WlDataDeviceManager, _: <WlDataDeviceManager as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlDataOffer, OfferData> for WorkerState {
    fn event(_: &mut Self, _: &WlDataOffer, event: wl_data_offer::Event, data: &OfferData, _: &Connection, _: &QueueHandle<Self>) {
        match event {
            wl_data_offer::Event::Offer { mime_type } => if let Ok(mut m) = data.mimes.lock() { m.push(mime_type); },
            wl_data_offer::Event::Action { dnd_action: WEnum::Value(action) } => if let Ok(mut a) = data.action.lock() { *a = action; },
            _ => {}
        }
    }
}

impl Dispatch<WlDataDevice, u32> for WorkerState {
    fn event(
        state: &mut Self,
        _: &WlDataDevice,
        event: wl_data_device::Event,
        &seat_name: &u32,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_device::Event::DataOffer { id } => {
                if let Some(dev) = state.devices.get_mut(&seat_name) {
                    dev.pending_offers.push(id);
                } else {
                    id.destroy();
                }
            }
            wl_data_device::Event::Enter { serial, surface, x, y, id } => {
                let surface_key = (surface.id().protocol_id(), surface.id().as_ptr() as usize);
                let matched = state
                    .windows
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .contains_key(&surface_key)
                    .then_some(surface_key);
                let has_offer = id.is_some();
                let has_uri_list = id
                    .as_ref()
                    .and_then(|o| o.data::<OfferData>())
                    .is_some_and(OfferData::has_uri_list);
                let prev_offer = if let Some(dev) = state.devices.get_mut(&seat_name) {
                    if let Some(offer) = id.as_ref() {
                        dev.pending_offers.retain(|o| o.id() != offer.id());
                    }
                    std::mem::replace(&mut dev.offer, id)
                } else {
                    if let Some(offer) = id {
                        offer.destroy();
                    }
                    return;
                };
                let drag_id = state.alloc_drag_id();
                state.step_seat(
                    seat_name,
                    DragEvent::Enter {
                        drag_id,
                        surface: matched,
                        serial,
                        at: (x as f32, y as f32),
                        has_uri_list,
                        has_offer,
                    },
                    prev_offer,
                );
            }
            wl_data_device::Event::Motion { x, y, .. } => {
                state.step_seat(seat_name, DragEvent::Motion { at: (x as f32, y as f32) }, None);
            }
            wl_data_device::Event::Leave => {
                state.step_seat(seat_name, DragEvent::Leave, None);
            }
            wl_data_device::Event::Drop => {
                let read_id = state.alloc_drag_id();
                state.step_seat(seat_name, DragEvent::Drop { read_id }, None);
            }
            wl_data_device::Event::Selection { id: Some(offer) } => {
                if let Some(dev) = state.devices.get_mut(&seat_name) {
                    dev.pending_offers.retain(|o| o.id() != offer.id());
                }
                // Clipboard reads/writes go through `arboard` (`wayland-data-control`);
                // destroy this device's selection offer immediately.
                offer.destroy();
            }
            _ => {}
        }
    }

    event_created_child!(WorkerState, WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (WlDataOffer, OfferData::default()),
    ]);
}
