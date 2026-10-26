//! CUDA Driver API bindings loaded from the system NVIDIA driver at runtime.
//! GPU kernels execute only as CUDA Graph nodes on an explicit non-blocking stream.
//! Legacy/default streams, direct standalone kernel launches and silent fallbacks
//! are forbidden. Host transfers also use this explicit stream.

/// Host-wall readback decomposition, not CUDA device-event timing.
/// The stream is already synchronized by GpuExecution::replay().
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaD2hTiming {
    pub enqueue: std::time::Duration,
    pub synchronize: std::time::Duration,
    pub host_copy: std::time::Duration,
    pub transfers: usize,
    pub bytes: usize,
}

#[cfg(target_os = "linux")]
mod platform {
    use std::collections::{BTreeMap, BTreeSet};
    use std::ffi::{CString, c_char, c_int, c_uint, c_void};
    use std::mem::transmute_copy;
    use std::ptr;

    use super::super::gpu::GpuError;
    use super::CudaD2hTiming;
    use std::time::Instant;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    const RTLD_NOW: c_int = 2;
    const CU_STREAM_NON_BLOCKING: c_uint = 1;
    const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
    const CU_EVENT_DEFAULT: c_uint = 0;

    // Runtime-only binary32 domain check; never advertised as a Lean proof.
    // Integer bit tests avoid FTZ and preserve the host's signed-zero rule.
    // The kernel launches exclusively during capture into a CUDA Graph.
    // A single CUDA Graph kernel checks one bucket of arbitrarily shaped
    // tensors. Entries are (device pointer, element count, domain mask) in
    // unsigned 64-bit words; all FP32 classification is bitwise.
    const DOMAIN_GUARD_PTX: &str = r#"
.version 7.0
.target sm_52
.address_size 64

.visible .entry vn_domain_guard(
    .param .u64 descriptors,
    .param .u64 flag,
    .param .u32 first
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<14>;
    ld.param.u64 %rd1, [descriptors];
    ld.param.u64 %rd2, [flag];
    ld.param.u32 %r1, [first];
    mov.u32 %r2, %ctaid.y;
    add.u32 %r3, %r1, %r2;
    mul.wide.u32 %rd3, %r3, 24;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u64 %rd5, [%rd4];
    add.u64 %rd6, %rd4, 8;
    ld.global.u64 %rd7, [%rd6];
    add.u64 %rd8, %rd4, 16;
    ld.global.u64 %rd9, [%rd8];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r7, %r5, %r6, %r4;
    cvt.u64.u32 %rd10, %r7;
    setp.ge.u64 %p1, %rd10, %rd7;
    @%p1 bra DONE;
    mul.wide.u32 %rd11, %r7, 4;
    add.u64 %rd12, %rd5, %rd11;
    ld.global.u32 %r8, [%rd12];
    and.b32 %r9, %r8, 0x7f800000;
    setp.eq.u32 %p2, %r9, 0x7f800000;
    and.b32 %r10, %r8, 0x80000000;
    and.b32 %r11, %r8, 0x7fffffff;
    setp.ne.u32 %p3, %r10, 0;
    setp.ne.u32 %p4, %r11, 0;
    and.pred %p5, %p3, %p4;
    setp.ne.u64 %p6, %rd9, 0;
    and.pred %p7, %p5, %p6;
    or.pred %p8, %p2, %p7;
    mov.u32 %r12, 1;
    @%p8 atom.global.or.b32 %r13, [%rd2], %r12;
DONE:
    ret;
}
"#;

    // An architecture-neutral transfer kernel: 2D grid's Y selects a
    // descriptor, X copies FP32 *bits* without arithmetic or conversion.
    // Only plans without read/write aliasing are permitted by the caller.
    const FEEDBACK_BATCH_PTX: &str = r#"
.version 7.0
.target sm_52
.address_size 64

.visible .entry vn_feedback_batch(
    .param .u64 descriptors,
    .param .u32 first
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<14>;
    ld.param.u64 %rd1, [descriptors];
    ld.param.u32 %r1, [first];
    mov.u32 %r2, %ctaid.y;
    add.u32 %r3, %r1, %r2;
    mul.wide.u32 %rd2, %r3, 24;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u64 %rd4, [%rd3];
    add.u64 %rd5, %rd3, 8;
    ld.global.u64 %rd6, [%rd5];
    add.u64 %rd7, %rd3, 16;
    ld.global.u64 %rd8, [%rd7];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r2, %r5, %r6, %r4;
    cvt.u64.u32 %rd9, %r2;
    setp.ge.u64 %p1, %rd9, %rd8;
    @%p1 bra DONE;
    mul.wide.u32 %rd10, %r2, 4;
    add.u64 %rd11, %rd6, %rd10;
    ld.global.u32 %r3, [%rd11];
    add.u64 %rd12, %rd4, %rd10;
    st.global.u32 [%rd12], %r3;
DONE:
    ret;
}
"#;

    // Bucket dissimilar sizes so tiny tensors never inherit the launch
    // width of the largest tensor. Groups are disjoint; starts index the
    // packed descriptor table. Kernel launches remain inside CUDA Graph.
    type TransferBucket = (u32, u32, u32);
    type TransferPlan = (Vec<usize>, Vec<TransferBucket>);

    fn transfer_buckets(lengths: &[usize]) -> Result<TransferPlan, GpuError> {
        let mut order = (0..lengths.len()).collect::<Vec<_>>();
        order.sort_by_key(|&i| lengths[i].next_power_of_two());
        let mut buckets = Vec::new();
        let mut first = 0;
        while first < order.len() {
            let key = lengths[order[first]].next_power_of_two();
            let mut end = first + 1;
            while end < order.len() && lengths[order[end]].next_power_of_two() == key {
                end += 1;
            }
            for start in (first..end).step_by(65_535) {
                let stop = end.min(start + 65_535);
                let biggest = lengths[order[stop - 1]];
                let x = u32::try_from(biggest.div_ceil(256))
                    .map_err(|_| GpuError::UnsupportedSize { elements: biggest })?;
                buckets.push((
                    u32::try_from(start)
                        .map_err(|_| GpuError::UnsupportedSize { elements: start })?,
                    u32::try_from(stop - start).map_err(|_| GpuError::UnsupportedSize {
                        elements: stop - start,
                    })?,
                    x,
                ));
            }
            first = end;
        }
        Ok((order, buckets))
    }

    fn descriptor_bytes(entries: &[[u64; 3]]) -> Vec<u8> {
        entries
            .iter()
            .flat_map(|triple| triple.iter().flat_map(|v| v.to_ne_bytes()))
            .collect()
    }

    struct Library(*mut c_void);

    impl Library {
        fn open() -> Result<Self, GpuError> {
            // The host driver, NOT the CUDA toolkit's stub library, provides
            // libcuda.so.1. NixOS exposes its kernel-matched driver under
            // /run/opengl-driver/lib, which may be absent from the dynamic
            // loader search path inside nix develop.
            for name in [c"libcuda.so.1", c"/run/opengl-driver/lib/libcuda.so.1"] {
                // SAFETY: each candidate is a NUL-terminated static string.
                let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
                if !handle.is_null() {
                    return Ok(Self(handle));
                }
            }
            Err(GpuError::Driver(
                "could not load host libcuda.so.1 (also tried /run/opengl-driver/lib/libcuda.so.1); check the NVIDIA driver and its library path".into(),
            ))
        }

        fn function<T: Copy>(&self, name: &'static str) -> Result<T, GpuError> {
            let cname = CString::new(name)
                .map_err(|_| GpuError::Driver("invalid CUDA symbol name".into()))?;
            // SAFETY: handle is live and cname is NUL-terminated.
            let symbol = unsafe { dlsym(self.0, cname.as_ptr()) };
            if symbol.is_null() {
                return Err(GpuError::Driver(format!("missing CUDA symbol {name}")));
            }
            if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
                return Err(GpuError::Driver(
                    "CUDA symbol has an invalid ABI size".into(),
                ));
            }
            // SAFETY: each caller chooses T to match the exact CUDA C ABI.
            Ok(unsafe { transmute_copy(&symbol) })
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: the handle was returned by dlopen and is owned by us.
            unsafe { dlclose(self.0) };
        }
    }

    type Handle = *mut c_void;
    type DevicePtr = u64;
    type ResultCode = c_int;

    struct Functions {
        init: unsafe extern "C" fn(c_uint) -> ResultCode,
        device_get: unsafe extern "C" fn(*mut c_int, c_int) -> ResultCode,
        primary_retain: unsafe extern "C" fn(*mut Handle, c_int) -> ResultCode,
        primary_release: unsafe extern "C" fn(c_int) -> ResultCode,
        context_push: unsafe extern "C" fn(Handle) -> ResultCode,
        context_pop: unsafe extern "C" fn(*mut Handle) -> ResultCode,
        module_load: unsafe extern "C" fn(*mut Handle, *const c_void) -> ResultCode,
        module_unload: unsafe extern "C" fn(Handle) -> ResultCode,
        module_function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> ResultCode,
        mem_allocate: unsafe extern "C" fn(*mut DevicePtr, usize) -> ResultCode,
        mem_free: unsafe extern "C" fn(DevicePtr) -> ResultCode,
        pinned_allocate: unsafe extern "C" fn(*mut *mut c_void, usize) -> ResultCode,
        pinned_free: unsafe extern "C" fn(*mut c_void) -> ResultCode,
        host_to_device: unsafe extern "C" fn(DevicePtr, *const c_void, usize, Handle) -> ResultCode,
        device_to_host: unsafe extern "C" fn(*mut c_void, DevicePtr, usize, Handle) -> ResultCode,
        device_to_device: unsafe extern "C" fn(DevicePtr, DevicePtr, usize, Handle) -> ResultCode,
        device_memset_u32: unsafe extern "C" fn(DevicePtr, c_uint, usize, Handle) -> ResultCode,
        stream_create: unsafe extern "C" fn(*mut Handle, c_uint) -> ResultCode,
        stream_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        stream_sync: unsafe extern "C" fn(Handle) -> ResultCode,
        stream_capture_begin: unsafe extern "C" fn(Handle, c_int) -> ResultCode,
        stream_capture_end: unsafe extern "C" fn(Handle, *mut Handle) -> ResultCode,
        graph_instantiate: unsafe extern "C" fn(*mut Handle, Handle, u64) -> ResultCode,
        graph_launch: unsafe extern "C" fn(Handle, Handle) -> ResultCode,
        graph_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        graph_exec_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        event_create: unsafe extern "C" fn(*mut Handle, c_uint) -> ResultCode,
        event_record: unsafe extern "C" fn(Handle, Handle) -> ResultCode,
        event_sync: unsafe extern "C" fn(Handle) -> ResultCode,
        event_elapsed: unsafe extern "C" fn(*mut f32, Handle, Handle) -> ResultCode,
        event_destroy: unsafe extern "C" fn(Handle) -> ResultCode,
        launch: unsafe extern "C" fn(
            Handle,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            Handle,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> ResultCode,
    }

    impl Functions {
        fn load(lib: &Library) -> Result<Self, GpuError> {
            Ok(Self {
                init: lib.function("cuInit")?,
                device_get: lib.function("cuDeviceGet")?,
                primary_retain: lib.function("cuDevicePrimaryCtxRetain")?,
                primary_release: lib.function("cuDevicePrimaryCtxRelease")?,
                context_push: lib.function("cuCtxPushCurrent_v2")?,
                context_pop: lib.function("cuCtxPopCurrent_v2")?,
                module_load: lib.function("cuModuleLoadData")?,
                module_unload: lib.function("cuModuleUnload")?,
                module_function: lib.function("cuModuleGetFunction")?,
                mem_allocate: lib.function("cuMemAlloc_v2")?,
                mem_free: lib.function("cuMemFree_v2")?,
                pinned_allocate: lib.function("cuMemAllocHost_v2")?,
                pinned_free: lib.function("cuMemFreeHost")?,
                host_to_device: lib.function("cuMemcpyHtoDAsync_v2")?,
                device_to_host: lib.function("cuMemcpyDtoHAsync_v2")?,
                device_to_device: lib.function("cuMemcpyDtoDAsync_v2")?,
                device_memset_u32: lib.function("cuMemsetD32Async")?,
                stream_create: lib.function("cuStreamCreate")?,
                stream_destroy: lib.function("cuStreamDestroy_v2")?,
                stream_sync: lib.function("cuStreamSynchronize")?,
                // CUDA Driver API: the unversioned cuStreamBeginCapture takes ONE
                // argument. We need the two-argument v2 ABI (mode included).
                stream_capture_begin: lib.function("cuStreamBeginCapture_v2")?,
                stream_capture_end: lib.function("cuStreamEndCapture")?,
                graph_instantiate: lib.function("cuGraphInstantiateWithFlags")?,
                graph_launch: lib.function("cuGraphLaunch")?,
                graph_destroy: lib.function("cuGraphDestroy")?,
                graph_exec_destroy: lib.function("cuGraphExecDestroy")?,
                event_create: lib.function("cuEventCreate")?,
                event_record: lib.function("cuEventRecord")?,
                event_sync: lib.function("cuEventSynchronize")?,
                event_elapsed: lib.function("cuEventElapsedTime")?,
                event_destroy: lib.function("cuEventDestroy_v2")?,
                launch: lib.function("cuLaunchKernel")?,
            })
        }
    }

    fn check(function: &'static str, code: ResultCode) -> Result<(), GpuError> {
        if code == 0 {
            Ok(())
        } else {
            Err(GpuError::Cuda { function, code })
        }
    }

    struct ContextGuard {
        pop: unsafe extern "C" fn(*mut Handle) -> ResultCode,
        expected: Handle,
    }

    impl Drop for ContextGuard {
        fn drop(&mut self) {
            let mut popped = ptr::null_mut();
            // SAFETY: this guard exists only after a successful push on this thread.
            let code = unsafe { (self.pop)(&mut popped) };
            debug_assert_eq!(code, 0);
            debug_assert_eq!(popped, self.expected);
        }
    }

    // A pinned allocation can be reused only after its previous async
    // transfer has completed on the session's explicit stream.
    struct PinnedBuffer {
        address: *mut c_void,
        capacity: usize,
    }

    /// Owns the stream, PTX module, device allocations and a retained reference
    /// to the shared primary CUDA context. Each session retains/releases once.
    /// The library is dropped last, after all CUDA handles are released.
    pub(crate) struct CudaSession {
        functions: Functions,
        device: c_int,
        context: Handle,
        module: Handle,
        allocations: Vec<(DevicePtr, usize)>,
        // Allocations are stable for the session lifetime. Keep O(log n)
        // preflight lookup while retaining reverse allocation-order teardown.
        allocation_capacity: BTreeMap<DevicePtr, usize>,
        staging_pending: Vec<PinnedBuffer>,
        staging_available: Vec<PinnedBuffer>,
        kernels: BTreeMap<String, Handle>,
        stream: Handle,
        graph: Handle,
        graph_exec: Handle,
        // Reusable CUDA Graph for stable feedback D2D copies, independent
        // from the main compute graph. No default stream or host staging.
        feedback_graph: Handle,
        feedback_graph_exec: Handle,
        feedback_plan: Vec<(DevicePtr, DevicePtr, usize)>,
        feedback_module: Handle,
        feedback_kernel: Handle,
        feedback_descriptors: DevicePtr,
        feedback_descriptor_capacity: usize,
        // A separate, reusable guard Graph checks all runtime-domain
        // obligations before any state-feedback copies are launched.
        guard_module: Handle,
        guard_kernel: Handle,
        guard_flag: DevicePtr,
        guard_graph: Handle,
        guard_graph_exec: Handle,
        guard_plan: Vec<(DevicePtr, usize, bool)>,
        guard_descriptors: DevicePtr,
        guard_descriptor_capacity: usize,
        capturing: bool,
        _library: Library,
    }

    impl CudaSession {
        fn enter_context(&self) -> Result<ContextGuard, GpuError> {
            // SAFETY: the context is owned by this session and remains live.
            check("cuCtxPushCurrent_v2", unsafe {
                (self.functions.context_push)(self.context)
            })?;
            Ok(ContextGuard {
                pop: self.functions.context_pop,
                expected: self.context,
            })
        }

        fn pop_context(&self) -> Result<(), GpuError> {
            let mut popped = ptr::null_mut();
            // SAFETY: callers only pop a context previously pushed on this thread.
            check("cuCtxPopCurrent_v2", unsafe {
                (self.functions.context_pop)(&mut popped)
            })?;
            if popped != self.context {
                return Err(GpuError::Driver(
                    "CUDA popped a different context than the session owns".into(),
                ));
            }
            Ok(())
        }

        pub fn new(ptx: &[u8]) -> Result<Self, GpuError> {
            let library = Library::open()?;
            let functions = Functions::load(&library)?;
            // SAFETY: CUDA has been dynamically loaded with matching ABI.
            check("cuInit", unsafe { (functions.init)(0) })?;
            let mut device = 0;
            // SAFETY: device is a valid writable pointer.
            check("cuDeviceGet", unsafe {
                (functions.device_get)(&mut device, 0)
            })?;
            // The primary context is shared by all sessions on this device.
            // Unlike cuCtxCreate, retain does not push it on this thread.
            let mut context = ptr::null_mut();
            check("cuDevicePrimaryCtxRetain", unsafe {
                (functions.primary_retain)(&mut context, device)
            })?;
            if context.is_null() {
                // A successful retain must be balanced even for a bad handle.
                unsafe { (functions.primary_release)(device) };
                return Err(GpuError::Driver(
                    "CUDA returned a null primary context".into(),
                ));
            }
            let mut session = Self {
                functions,
                device,
                context,
                module: ptr::null_mut(),
                allocations: Vec::new(),
                allocation_capacity: BTreeMap::new(),
                staging_pending: Vec::new(),
                staging_available: Vec::new(),
                kernels: BTreeMap::new(),
                stream: ptr::null_mut(),
                graph: ptr::null_mut(),
                graph_exec: ptr::null_mut(),
                feedback_graph: ptr::null_mut(),
                feedback_graph_exec: ptr::null_mut(),
                feedback_plan: Vec::new(),
                feedback_module: ptr::null_mut(),
                feedback_kernel: ptr::null_mut(),
                feedback_descriptors: 0,
                feedback_descriptor_capacity: 0,
                guard_module: ptr::null_mut(),
                guard_kernel: ptr::null_mut(),
                guard_flag: 0,
                guard_graph: ptr::null_mut(),
                guard_graph_exec: ptr::null_mut(),
                guard_plan: Vec::new(),
                guard_descriptors: 0,
                guard_descriptor_capacity: 0,
                capturing: false,
                _library: library,
            };
            // This guard is dropped before session teardown on any error.
            let _context = session.enter_context()?;
            let bytes = CString::new(ptx)
                .map_err(|_| GpuError::Driver("PTX contains interior NUL byte".into()))?;
            // SAFETY: PTX has an extra terminating NUL and lives through the call.
            check("cuModuleLoadData", unsafe {
                (session.functions.module_load)(
                    &mut session.module,
                    bytes.as_ptr().cast::<c_void>(),
                )
            })?;
            // Never pass 0/CU_STREAM_LEGACY/CU_STREAM_PER_THREAD as a launch stream.
            // No fallback: require an explicitly allocated non-blocking stream.
            check("cuStreamCreate", unsafe {
                (session.functions.stream_create)(&mut session.stream, CU_STREAM_NON_BLOCKING)
            })?;
            if session.stream.is_null() {
                return Err(GpuError::Driver(
                    "CUDA returned a null stream handle".into(),
                ));
            }

            // _context pops the retained primary context after setup. Sessions
            // share a device context but never share stream/module ownership.
            Ok(session)
        }

        pub fn allocate(&mut self, elements: usize) -> Result<DevicePtr, GpuError> {
            let _context = self.enter_context()?;
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(GpuError::UnsupportedSize { elements })?;
            let mut address = 0;
            // SAFETY: address is writable, CUDA manages allocated device memory.
            check("cuMemAlloc_v2", unsafe {
                (self.functions.mem_allocate)(&mut address, bytes)
            })?;
            self.allocations.push((address, elements));
            self.allocation_capacity.insert(address, elements);
            Ok(address)
        }

        /// Check the actual device allocations backing a logical memory
        /// contract before graph capture. This is a preflight check, not an
        /// assertion about LLVM/PTX's machine-level memory accesses.
        pub fn require_allocation(
            &self,
            address: DevicePtr,
            elements: usize,
        ) -> Result<(), GpuError> {
            if self
                .allocation_capacity
                .get(&address)
                .is_none_or(|&capacity| capacity < elements)
            {
                return Err(GpuError::Driver(
                    "CUDA kernel memory contract references an unallocated or undersized buffer"
                        .into(),
                ));
            }
            Ok(())
        }

        // Find the smallest completed allocation that fits, or allocate one.
        // Never recycle in-flight buffers: async CUDA copies still read them.
        fn staging_buffer(&mut self, bytes: usize) -> Result<PinnedBuffer, GpuError> {
            let best = self
                .staging_available
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.capacity >= bytes)
                .min_by_key(|(_, entry)| entry.capacity)
                .map(|(index, _)| index);
            if let Some(index) = best {
                return Ok(self.staging_available.swap_remove(index));
            }
            let mut address = ptr::null_mut();
            check("cuMemAllocHost_v2", unsafe {
                (self.functions.pinned_allocate)(&mut address, bytes)
            })?;
            if address.is_null() {
                return Err(GpuError::Driver(
                    "CUDA returned null pinned allocation".into(),
                ));
            }
            Ok(PinnedBuffer {
                address,
                capacity: bytes,
            })
        }

        // Packed metadata is never interpreted as f32. The transfer is
        // staged in pinned host memory until the non-blocking stream sync.
        fn upload_descriptor_bytes(
            &mut self,
            address: DevicePtr,
            bytes: &[u8],
        ) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            let staging = self.staging_buffer(bytes.len())?;
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), staging.address.cast::<u8>(), bytes.len())
            };
            let pinned = staging.address;
            self.staging_pending.push(staging);
            check("cuMemcpyHtoDAsync_v2", unsafe {
                (self.functions.host_to_device)(address, pinned, bytes.len(), self.stream)
            })
        }

        /// Stage input bytes in page-locked memory and enqueue on our explicit
        /// non-blocking stream. No buffer is recycled before synchronization.
        pub fn copy_to_device(
            &mut self,
            address: DevicePtr,
            values: &[f32],
        ) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            let bytes = std::mem::size_of_val(values);
            let staging = self.staging_buffer(bytes)?;
            // SAFETY: staging has capacity for values.len() f32s.
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    staging.address.cast::<f32>(),
                    values.len(),
                )
            };
            // Keep the staging allocation alive even if enqueueing fails:
            // CUDA may have observed it before returning an error.
            let pinned = staging.address;
            self.staging_pending.push(staging);
            check("cuMemcpyHtoDAsync_v2", unsafe {
                (self.functions.host_to_device)(address, pinned, bytes, self.stream)
            })
        }

        /// State feedback happens on the existing explicit non-blocking stream.
        /// It is not part of graph capture and never round-trips through the host.
        pub fn copy_device_to_device(
            &mut self,
            destination: DevicePtr,
            source: DevicePtr,
            elements: usize,
        ) -> Result<(), GpuError> {
            self.require_allocation(destination, elements)?;
            self.require_allocation(source, elements)?;
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(GpuError::UnsupportedSize { elements })?;
            let _context = self.enter_context()?;
            check("cuMemcpyDtoDAsync_v2", unsafe {
                (self.functions.device_to_device)(destination, source, bytes, self.stream)
            })
        }

        /// Check actual output domains on GPU and read only one status word.
        /// False means at least one FP32 NaN, Infinity, or (if requested)
        /// strictly negative nonzero value. Negative zero is accepted.
        /// The status is initialized before *every* replay. This is a
        /// runtime hardware check, not a static proof or semantic-IR kernel.
        pub fn validate_output_domains(
            &mut self,
            sources: &[(DevicePtr, usize, bool)],
        ) -> Result<bool, GpuError> {
            if sources.is_empty() {
                return Ok(true);
            }
            if self.capturing {
                return Err(GpuError::Driver(
                    "cannot validate feedback while CUDA Graph capture is active".into(),
                ));
            }
            // A published graph owns an immutable descriptor table, and
            // allocations cannot be freed before CudaSession teardown.
            // Only a changed plan needs another full host preflight.
            let rebuild = self.guard_plan.as_slice() != sources || self.guard_graph_exec.is_null();
            if rebuild {
                for &(address, elements, _) in sources {
                    self.require_allocation(address, elements)?;
                    if elements == 0 || elements > i32::MAX as usize {
                        return Err(GpuError::UnsupportedSize { elements });
                    }
                }
            }
            let _context = self.enter_context()?;
            if self.guard_module.is_null() {
                let source = CString::new(DOMAIN_GUARD_PTX)
                    .map_err(|_| GpuError::Driver("domain guard PTX contains NUL".into()))?;
                check("cuModuleLoadData", unsafe {
                    (self.functions.module_load)(
                        &mut self.guard_module,
                        source.as_ptr().cast::<c_void>(),
                    )
                })?;
                check("cuModuleGetFunction", unsafe {
                    (self.functions.module_function)(
                        &mut self.guard_kernel,
                        self.guard_module,
                        c"vn_domain_guard".as_ptr(),
                    )
                })?;
                self.guard_flag = self.allocate(1)?;
            }
            if rebuild {
                self.synchronize()?;
                if !self.guard_graph_exec.is_null() {
                    check("cuGraphExecDestroy", unsafe {
                        (self.functions.graph_exec_destroy)(self.guard_graph_exec)
                    })?;
                    self.guard_graph_exec = ptr::null_mut();
                }
                if !self.guard_graph.is_null() {
                    check("cuGraphDestroy", unsafe {
                        (self.functions.graph_destroy)(self.guard_graph)
                    })?;
                    self.guard_graph = ptr::null_mut();
                }
                self.guard_plan.clear();
                let (order, batches) =
                    transfer_buckets(&sources.iter().map(|source| source.1).collect::<Vec<_>>())?;
                let entries = order
                    .iter()
                    .map(|&index| {
                        let (address, elements, nonnegative) = sources[index];
                        [address, elements as u64, u64::from(nonnegative)]
                    })
                    .collect::<Vec<_>>();
                let bytes = descriptor_bytes(&entries);
                let words = bytes.len() / std::mem::size_of::<f32>();
                if words > self.guard_descriptor_capacity {
                    self.guard_descriptors = self.allocate(words)?;
                    self.guard_descriptor_capacity = words;
                }
                self.upload_descriptor_bytes(self.guard_descriptors, &bytes)?;
                // A fresh immutable descriptor table must outlive every
                // capture and replay that addresses it.
                self.synchronize()?;
                check("cuStreamBeginCapture_v2", unsafe {
                    (self.functions.stream_capture_begin)(
                        self.stream,
                        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                })?;
                self.capturing = true;
                for (start, count, grid) in batches {
                    let mut descriptors = self.guard_descriptors;
                    let mut flag = self.guard_flag;
                    let mut first = start;
                    let mut args = [
                        (&mut descriptors as *mut DevicePtr).cast::<c_void>(),
                        (&mut flag as *mut DevicePtr).cast::<c_void>(),
                        (&mut first as *mut u32).cast::<c_void>(),
                    ];
                    check("cuLaunchKernel", unsafe {
                        (self.functions.launch)(
                            self.guard_kernel,
                            grid,
                            count,
                            1,
                            256,
                            1,
                            1,
                            0,
                            self.stream,
                            args.as_mut_ptr(),
                            ptr::null_mut(),
                        )
                    })?;
                }
                let mut graph = ptr::null_mut();
                let ended = check("cuStreamEndCapture", unsafe {
                    (self.functions.stream_capture_end)(self.stream, &mut graph)
                });
                self.capturing = false;
                self.guard_graph = graph;
                ended?;
                if self.guard_graph.is_null() {
                    return Err(GpuError::Driver(
                        "domain guard CUDA Graph capture returned null".into(),
                    ));
                }
                check("cuGraphInstantiateWithFlags", unsafe {
                    (self.functions.graph_instantiate)(
                        &mut self.guard_graph_exec,
                        self.guard_graph,
                        0,
                    )
                })?;
                self.guard_plan.extend_from_slice(sources);
            }
            check("cuMemsetD32Async", unsafe {
                (self.functions.device_memset_u32)(self.guard_flag, 0, 1, self.stream)
            })?;
            check("cuGraphLaunch", unsafe {
                (self.functions.graph_launch)(self.guard_graph_exec, self.stream)
            })?;
            let mut result = [0i32; 1];
            self.copy_indices_to_host(&mut result, self.guard_flag)?;
            Ok(result[0] == 0)
        }

        /// Capture the checked stable device-feedback copy plan once and
        /// replay it on the explicit non-blocking stream. A change in bindings
        /// invalidates the cache and is recaptured after synchronization.
        /// This is a *transfer* graph, not a bypass around the domain gate.
        pub fn launch_feedback_copies(
            &mut self,
            copies: &[(DevicePtr, DevicePtr, usize)],
        ) -> Result<(), GpuError> {
            if copies.is_empty() {
                return Ok(());
            }
            if self.capturing {
                return Err(GpuError::Driver(
                    "feedback is forbidden during CUDA Graph capture".into(),
                ));
            }
            let rebuild =
                self.feedback_plan.as_slice() != copies || self.feedback_graph_exec.is_null();
            if rebuild {
                // The cached graph retains its checked descriptor table for
                // the session lifetime. Changing even one pointer/extent
                // requires fresh size and simultaneous-assignment checks.
                for &(dest, src, elements) in copies {
                    self.require_allocation(dest, elements)?;
                    self.require_allocation(src, elements)?;
                    if elements == 0 || elements > i32::MAX as usize {
                        return Err(GpuError::UnsupportedSize { elements });
                    }
                }
                let destinations = copies
                    .iter()
                    .map(|&(dst, _, _)| dst)
                    .collect::<BTreeSet<_>>();
                if destinations.len() != copies.len()
                    || copies
                        .iter()
                        .any(|&(dst, src, _)| dst != src && destinations.contains(&src))
                {
                    return Err(GpuError::InvalidInput {
                        node: 0,
                        reason: "grouped feedback destinations overlap another transfer",
                    });
                }
            }
            let _context = self.enter_context()?;
            if self.feedback_module.is_null() {
                let source = CString::new(FEEDBACK_BATCH_PTX)
                    .map_err(|_| GpuError::Driver("feedback kernel PTX contains NUL".into()))?;
                check("cuModuleLoadData", unsafe {
                    (self.functions.module_load)(
                        &mut self.feedback_module,
                        source.as_ptr().cast::<c_void>(),
                    )
                })?;
                check("cuModuleGetFunction", unsafe {
                    (self.functions.module_function)(
                        &mut self.feedback_kernel,
                        self.feedback_module,
                        c"vn_feedback_batch".as_ptr(),
                    )
                })?;
            }
            if rebuild {
                self.synchronize()?;
                if !self.feedback_graph_exec.is_null() {
                    check("cuGraphExecDestroy", unsafe {
                        (self.functions.graph_exec_destroy)(self.feedback_graph_exec)
                    })?;
                    self.feedback_graph_exec = ptr::null_mut();
                }
                if !self.feedback_graph.is_null() {
                    check("cuGraphDestroy", unsafe {
                        (self.functions.graph_destroy)(self.feedback_graph)
                    })?;
                    self.feedback_graph = ptr::null_mut();
                }
                self.feedback_plan.clear();
                let (order, batches) =
                    transfer_buckets(&copies.iter().map(|copy| copy.2).collect::<Vec<_>>())?;
                let entries = order
                    .iter()
                    .map(|&index| {
                        let (dest, source, elements) = copies[index];
                        [dest, source, elements as u64]
                    })
                    .collect::<Vec<_>>();
                let bytes = descriptor_bytes(&entries);
                let words = bytes.len() / std::mem::size_of::<f32>();
                if words > self.feedback_descriptor_capacity {
                    self.feedback_descriptors = self.allocate(words)?;
                    self.feedback_descriptor_capacity = words;
                }
                self.upload_descriptor_bytes(self.feedback_descriptors, &bytes)?;
                self.synchronize()?;
                check("cuStreamBeginCapture_v2", unsafe {
                    (self.functions.stream_capture_begin)(
                        self.stream,
                        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                })?;
                self.capturing = true;
                for (start, count, grid) in batches {
                    let mut descriptors = self.feedback_descriptors;
                    let mut first = start;
                    let mut args = [
                        (&mut descriptors as *mut DevicePtr).cast::<c_void>(),
                        (&mut first as *mut u32).cast::<c_void>(),
                    ];
                    check("cuLaunchKernel", unsafe {
                        (self.functions.launch)(
                            self.feedback_kernel,
                            grid,
                            count,
                            1,
                            256,
                            1,
                            1,
                            0,
                            self.stream,
                            args.as_mut_ptr(),
                            ptr::null_mut(),
                        )
                    })?;
                }
                let mut graph = ptr::null_mut();
                let ended = check("cuStreamEndCapture", unsafe {
                    (self.functions.stream_capture_end)(self.stream, &mut graph)
                });
                self.capturing = false;
                self.feedback_graph = graph;
                ended?;
                if self.feedback_graph.is_null() {
                    return Err(GpuError::Driver(
                        "feedback CUDA Graph capture returned null".into(),
                    ));
                }
                check("cuGraphInstantiateWithFlags", unsafe {
                    (self.functions.graph_instantiate)(
                        &mut self.feedback_graph_exec,
                        self.feedback_graph,
                        0,
                    )
                })?;
                self.feedback_plan.extend_from_slice(copies);
            }
            check("cuGraphLaunch", unsafe {
                (self.functions.graph_launch)(self.feedback_graph_exec, self.stream)
            })
        }

        /// Opt-in CUDA event measurement of the *same* grouped D2D graph.
        /// This does not replay transfers a second time; it changes the
        /// caller's synchronization point and is unsuitable for throughput
        /// benchmarking. Events use the owned non-blocking stream only.
        pub fn launch_feedback_copies_device_timed(
            &mut self,
            copies: &[(DevicePtr, DevicePtr, usize)],
        ) -> Result<std::time::Duration, GpuError> {
            if copies.is_empty() {
                return Ok(std::time::Duration::ZERO);
            }
            let _context = self.enter_context()?;
            let mut start = ptr::null_mut();
            check("cuEventCreate", unsafe {
                (self.functions.event_create)(&mut start, CU_EVENT_DEFAULT)
            })?;
            let mut stop = ptr::null_mut();
            if let Err(err) = check("cuEventCreate", unsafe {
                (self.functions.event_create)(&mut stop, CU_EVENT_DEFAULT)
            }) {
                unsafe { (self.functions.event_destroy)(start) };
                return Err(err);
            }
            let result = (|| {
                check("cuEventRecord", unsafe {
                    (self.functions.event_record)(start, self.stream)
                })?;
                self.launch_feedback_copies(copies)?;
                check("cuEventRecord", unsafe {
                    (self.functions.event_record)(stop, self.stream)
                })?;
                check("cuEventSynchronize", unsafe {
                    (self.functions.event_sync)(stop)
                })?;
                let mut elapsed_ms = 0.0f32;
                check("cuEventElapsedTime", unsafe {
                    (self.functions.event_elapsed)(&mut elapsed_ms, start, stop)
                })?;
                Ok(std::time::Duration::from_secs_f64(
                    f64::from(elapsed_ms) / 1000.0,
                ))
            })();
            unsafe {
                (self.functions.event_destroy)(stop);
                (self.functions.event_destroy)(start);
            }
            result
        }

        fn copy_to_host_bytes(
            &mut self,
            out: *mut c_void,
            bytes: usize,
            address: DevicePtr,
        ) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            let staging = self.staging_buffer(bytes)?;
            let pinned = staging.address;
            self.staging_pending.push(staging);
            // SAFETY: pinned staging remains live until synchronization completes.
            check("cuMemcpyDtoHAsync_v2", unsafe {
                (self.functions.device_to_host)(pinned, address, bytes, self.stream)
            })?;
            self.synchronize()?;
            // SAFETY: the async transfer completed and the caller supplied a
            // writable destination covering exactly `bytes`.
            unsafe { ptr::copy_nonoverlapping(pinned.cast::<u8>(), out.cast::<u8>(), bytes) };
            self.release_completed_staging();
            Ok(())
        }

        /// Queue F32 device-to-host copies into session-owned pinned staging
        /// on the same explicit non-blocking CUDA stream. Synchronize once,
        /// then copy the completed data into ordinary owned host vectors.
        /// Pending pinned buffers remain alive on any enqueue/sync failure,
        /// and are reclaimed only after a successful stream barrier.
        pub fn copy_f32_batch_to_host(
            &mut self,
            transfers: &[(DevicePtr, usize)],
        ) -> Result<Vec<Vec<f32>>, GpuError> {
            self.copy_f32_batch_to_host_profiled(transfers)
                .map(|(values, _)| values)
        }

        pub fn copy_f32_batch_to_host_profiled(
            &mut self,
            transfers: &[(DevicePtr, usize)],
        ) -> Result<(Vec<Vec<f32>>, CudaD2hTiming), GpuError> {
            if transfers.is_empty() {
                return Ok((Vec::new(), CudaD2hTiming::default()));
            }
            // Validate the complete batch before queuing any DMA.
            let byte_counts = transfers
                .iter()
                .map(|&(_, elements)| {
                    elements
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or(GpuError::UnsupportedSize { elements })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let bytes = byte_counts.iter().try_fold(0usize, |sum, &count| {
                sum.checked_add(count)
                    .ok_or(GpuError::UnsupportedSize { elements: sum })
            })?;
            let enqueue_started = Instant::now();
            let _context = self.enter_context()?;
            let mut staged = Vec::with_capacity(transfers.len());
            for (&(address, elements), &bytes) in transfers.iter().zip(&byte_counts) {
                let pinned = self.staging_buffer(bytes)?;
                let pointer = pinned.address;
                self.staging_pending.push(pinned);
                check("cuMemcpyDtoHAsync_v2", unsafe {
                    (self.functions.device_to_host)(pointer, address, bytes, self.stream)
                })?;
                staged.push((pointer, elements));
            }
            let enqueue = enqueue_started.elapsed();
            let sync_started = Instant::now();
            self.synchronize()?;
            let synchronize = sync_started.elapsed();
            let copy_started = Instant::now();
            let mut outputs = Vec::with_capacity(staged.len());
            for (pointer, elements) in staged {
                let mut values = vec![0f32; elements];
                // SAFETY: the copy completed on this stream; staging is
                // pinned and remains owned by staging_pending until release.
                unsafe {
                    ptr::copy_nonoverlapping(pointer.cast::<f32>(), values.as_mut_ptr(), elements)
                };
                outputs.push(values);
            }
            self.release_completed_staging();
            Ok((
                outputs,
                CudaD2hTiming {
                    enqueue,
                    synchronize,
                    host_copy: copy_started.elapsed(),
                    transfers: transfers.len(),
                    bytes,
                },
            ))
        }

        pub fn copy_to_host(
            &mut self,
            out: &mut [f32],
            address: DevicePtr,
        ) -> Result<(), GpuError> {
            self.copy_to_host_bytes(
                out.as_mut_ptr().cast::<c_void>(),
                std::mem::size_of_val(out),
                address,
            )
        }

        pub fn copy_indices_to_host(
            &mut self,
            out: &mut [i32],
            address: DevicePtr,
        ) -> Result<(), GpuError> {
            self.copy_to_host_bytes(
                out.as_mut_ptr().cast::<c_void>(),
                std::mem::size_of_val(out),
                address,
            )
        }

        // Reuse completed staging buffers instead of allocating/freeing pinned
        // memory on every update_inputs/read_output. The session owns both lists.
        fn release_completed_staging(&mut self) {
            self.staging_available.append(&mut self.staging_pending);
        }

        pub fn flush_input_copies(&mut self) -> Result<(), GpuError> {
            self.synchronize()?;
            self.release_completed_staging();
            Ok(())
        }

        /// Begin capturing only kernel launches to a CUDA Graph. No allocations,
        /// host copies, or default-stream commands are permitted during capture.
        pub fn begin_graph(&mut self) -> Result<(), GpuError> {
            if self.capturing || !self.graph.is_null() || !self.graph_exec.is_null() {
                return Err(GpuError::Driver("CUDA Graph already in progress".into()));
            }
            self.flush_input_copies()?;
            // Keep this session context current for the entire capture sequence.
            check("cuCtxPushCurrent_v2", unsafe {
                (self.functions.context_push)(self.context)
            })?;
            // SAFETY: an explicitly created nonblocking stream is current.
            let code = unsafe {
                (self.functions.stream_capture_begin)(
                    self.stream,
                    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            };
            if let Err(err) = check("cuStreamBeginCapture", code) {
                let _ = self.pop_context();
                return Err(err);
            }
            self.capturing = true;
            Ok(())
        }

        /// Finish capture and instantiate an executable graph without launching it.
        pub fn finish_graph(&mut self) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver("CUDA Graph was not started".into()));
            }
            let result = (|| {
                let mut graph = ptr::null_mut();
                // SAFETY: capture runs on the current thread and same CUDA stream.
                let code = unsafe { (self.functions.stream_capture_end)(self.stream, &mut graph) };
                self.capturing = false;
                self.graph = graph;
                check("cuStreamEndCapture", code)?;
                if self.graph.is_null() {
                    return Err(GpuError::Driver("CUDA Graph capture returned null".into()));
                }
                // SAFETY: all captured kernel parameters have been copied by CUDA.
                check("cuGraphInstantiateWithFlags", unsafe {
                    (self.functions.graph_instantiate)(&mut self.graph_exec, self.graph, 0)
                })
            })();
            let pop_result = self.pop_context();
            result?;
            pop_result
        }

        /// Replay an already instantiated graph on the owned non-blocking stream.
        pub fn launch_graph(&self) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            self.launch_graph_current_context()
        }

        // For use inside a longer-lived context guard (CUDA-event benchmarks).
        fn launch_graph_current_context(&self) -> Result<(), GpuError> {
            if self.capturing || self.graph_exec.is_null() {
                return Err(GpuError::Driver(
                    "CUDA Graph must be instantiated before replay".into(),
                ));
            }
            check("cuGraphLaunch", unsafe {
                (self.functions.graph_launch)(self.graph_exec, self.stream)
            })
        }

        /// Measure graph replay on the CUDA stream with device events.
        pub fn benchmark_graph(
            &self,
            warmup_launches: usize,
            measured_launches: usize,
        ) -> Result<f32, GpuError> {
            let _context = self.enter_context()?;
            for _ in 0..warmup_launches {
                self.launch_graph_current_context()?;
            }
            self.synchronize()?;

            let mut start = ptr::null_mut();
            check("cuEventCreate", unsafe {
                (self.functions.event_create)(&mut start, CU_EVENT_DEFAULT)
            })?;
            let mut stop = ptr::null_mut();
            if let Err(err) = check("cuEventCreate", unsafe {
                (self.functions.event_create)(&mut stop, CU_EVENT_DEFAULT)
            }) {
                unsafe { (self.functions.event_destroy)(start) };
                return Err(err);
            }

            let result = (|| {
                check("cuEventRecord", unsafe {
                    (self.functions.event_record)(start, self.stream)
                })?;
                for _ in 0..measured_launches {
                    self.launch_graph_current_context()?;
                }
                check("cuEventRecord", unsafe {
                    (self.functions.event_record)(stop, self.stream)
                })?;
                check("cuEventSynchronize", unsafe {
                    (self.functions.event_sync)(stop)
                })?;
                let mut milliseconds = 0.0f32;
                check("cuEventElapsedTime", unsafe {
                    (self.functions.event_elapsed)(&mut milliseconds, start, stop)
                })?;
                Ok(milliseconds)
            })();

            unsafe {
                (self.functions.event_destroy)(stop);
                (self.functions.event_destroy)(start);
            }
            result
        }

        /// Resolve the CUDA function before capture. Lazy lookup/JIT is not
        /// part of the captured stream's execution dependency graph.
        pub fn prepare_kernel(&mut self, name: &str) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            if self.capturing {
                return Err(GpuError::Driver(
                    "kernel resolution during CUDA Graph capture is forbidden".into(),
                ));
            }
            if self.kernels.contains_key(name) {
                return Ok(());
            }
            let c_name = CString::new(name)
                .map_err(|_| GpuError::Driver("kernel name contains NUL".into()))?;
            let mut function = ptr::null_mut();
            // SAFETY: module is loaded and the C string is NUL terminated.
            check("cuModuleGetFunction", unsafe {
                (self.functions.module_function)(&mut function, self.module, c_name.as_ptr())
            })?;
            self.kernels.insert(name.to_owned(), function);
            Ok(())
        }

        pub fn launch(
            &self,
            name: &str,
            left: DevicePtr,
            right: DevicePtr,
            output: DevicePtr,
            length: usize,
            threads: u32,
        ) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver(
                    "direct kernel launch is forbidden; capture a CUDA Graph".into(),
                ));
            }
            let function = *self.kernels.get(name).ok_or_else(|| {
                GpuError::Driver(
                    "kernel function must be resolved before CUDA Graph capture".into(),
                )
            })?;
            let count = i32::try_from(length)
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let grid = u32::try_from(length.div_ceil(threads as usize))
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let mut lhs = left;
            let mut rhs = right;
            let mut dst = output;
            let mut n = count;
            let mut args = [
                (&mut lhs as *mut DevicePtr).cast::<c_void>(),
                (&mut rhs as *mut DevicePtr).cast::<c_void>(),
                (&mut dst as *mut DevicePtr).cast::<c_void>(),
                (&mut n as *mut i32).cast::<c_void>(),
            ];
            // SAFETY: each kernel argument points to correctly sized stable
            // storage matching the LLVM kernel signature. The launch itself
            // is asynchronous; synchronize before copying results.
            check("cuLaunchKernel", unsafe {
                (self.functions.launch)(
                    function,
                    grid,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })
        }

        pub fn launch3(
            &self,
            name: &str,
            sources: [DevicePtr; 3],
            output: DevicePtr,
            length: usize,
            threads: u32,
        ) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver(
                    "direct kernel launch is forbidden; capture a CUDA Graph".into(),
                ));
            }
            let function = *self.kernels.get(name).ok_or_else(|| {
                GpuError::Driver(
                    "kernel function must be resolved before CUDA Graph capture".into(),
                )
            })?;
            let count = i32::try_from(length)
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let grid = u32::try_from(length.div_ceil(threads as usize))
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let [mut a, mut b, mut c] = sources;
            let mut dst = output;
            let mut n = count;
            let mut args = [
                (&mut a as *mut DevicePtr).cast::<c_void>(),
                (&mut b as *mut DevicePtr).cast::<c_void>(),
                (&mut c as *mut DevicePtr).cast::<c_void>(),
                (&mut dst as *mut DevicePtr).cast::<c_void>(),
                (&mut n as *mut i32).cast::<c_void>(),
            ];
            check("cuLaunchKernel", unsafe {
                (self.functions.launch)(
                    function,
                    grid,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })
        }

        /// Four-source kernel capture. The sole launch is performed on the
        /// session-owned CU_STREAM_NON_BLOCKING while a CUDA Graph is captured.
        pub fn launch4(
            &self,
            name: &str,
            sources: [DevicePtr; 4],
            output: DevicePtr,
            length: usize,
            threads: u32,
        ) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver(
                    "direct kernel launch is forbidden; capture a CUDA Graph".into(),
                ));
            }
            let function = *self.kernels.get(name).ok_or_else(|| {
                GpuError::Driver(
                    "kernel function must be resolved before CUDA Graph capture".into(),
                )
            })?;
            let count = i32::try_from(length)
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let grid = u32::try_from(length.div_ceil(threads as usize))
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let [mut a, mut b, mut c, mut d] = sources;
            let mut dst = output;
            let mut n = count;
            let mut args = [
                (&mut a as *mut DevicePtr).cast::<c_void>(),
                (&mut b as *mut DevicePtr).cast::<c_void>(),
                (&mut c as *mut DevicePtr).cast::<c_void>(),
                (&mut d as *mut DevicePtr).cast::<c_void>(),
                (&mut dst as *mut DevicePtr).cast::<c_void>(),
                (&mut n as *mut i32).cast::<c_void>(),
            ];
            check("cuLaunchKernel", unsafe {
                (self.functions.launch)(
                    function,
                    grid,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })
        }

        /// Capture a bounded two-output pointwise kernel on the owned
        /// non-blocking CUDA stream. Neither output aliases any source.
        pub fn launch4_dual(
            &self,
            name: &str,
            sources: [DevicePtr; 4],
            outputs: [DevicePtr; 2],
            length: usize,
            threads: u32,
        ) -> Result<(), GpuError> {
            if !self.capturing {
                return Err(GpuError::Driver(
                    "direct kernel launch is forbidden; capture a CUDA Graph".into(),
                ));
            }
            let function = *self.kernels.get(name).ok_or_else(|| {
                GpuError::Driver(
                    "kernel function must be resolved before CUDA Graph capture".into(),
                )
            })?;
            let count = i32::try_from(length)
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let grid = u32::try_from(length.div_ceil(threads as usize))
                .map_err(|_| GpuError::UnsupportedSize { elements: length })?;
            let [mut a, mut b, mut c, mut d] = sources;
            let [mut extra, mut dst] = outputs;
            let mut n = count;
            let mut args = [
                (&mut a as *mut DevicePtr).cast::<c_void>(),
                (&mut b as *mut DevicePtr).cast::<c_void>(),
                (&mut c as *mut DevicePtr).cast::<c_void>(),
                (&mut d as *mut DevicePtr).cast::<c_void>(),
                (&mut extra as *mut DevicePtr).cast::<c_void>(),
                (&mut dst as *mut DevicePtr).cast::<c_void>(),
                (&mut n as *mut i32).cast::<c_void>(),
            ];
            check("cuLaunchKernel", unsafe {
                (self.functions.launch)(
                    function,
                    grid,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })
        }

        pub fn synchronize(&self) -> Result<(), GpuError> {
            let _context = self.enter_context()?;
            // SAFETY: stream belongs to this CUDA context; never synchronize legacy.
            check("cuStreamSynchronize", unsafe {
                (self.functions.stream_sync)(self.stream)
            })
        }
    }

    impl Drop for CudaSession {
        fn drop(&mut self) {
            // begin_graph intentionally keeps the context pushed until capture ends.
            let mut context_pushed = self.capturing;
            if !context_pushed && !self.context.is_null() {
                context_pushed = unsafe { (self.functions.context_push)(self.context) } == 0;
            }

            // The stream must complete before releasing graph/module/device/pinned
            // buffers. A failed capture is ended best-effort without replay.
            if self.capturing && !self.stream.is_null() {
                let mut graph = ptr::null_mut();
                // SAFETY: same CUDA stream, no concurrent host access.
                unsafe { (self.functions.stream_capture_end)(self.stream, &mut graph) };
                self.capturing = false;
                if !graph.is_null() {
                    unsafe { (self.functions.graph_destroy)(graph) };
                }
            }
            if !self.stream.is_null() {
                // SAFETY: session owns this stream.
                unsafe { (self.functions.stream_sync)(self.stream) };
            }
            if !self.guard_graph_exec.is_null() {
                unsafe { (self.functions.graph_exec_destroy)(self.guard_graph_exec) };
            }
            if !self.guard_graph.is_null() {
                unsafe { (self.functions.graph_destroy)(self.guard_graph) };
            }
            if !self.feedback_graph_exec.is_null() {
                unsafe { (self.functions.graph_exec_destroy)(self.feedback_graph_exec) };
            }
            if !self.feedback_graph.is_null() {
                unsafe { (self.functions.graph_destroy)(self.feedback_graph) };
            }
            if !self.graph_exec.is_null() {
                // SAFETY: the executable graph is owned by this session.
                unsafe { (self.functions.graph_exec_destroy)(self.graph_exec) };
            }
            if !self.graph.is_null() {
                unsafe { (self.functions.graph_destroy)(self.graph) };
            }
            for (address, _) in self.allocations.drain(..).rev() {
                unsafe { (self.functions.mem_free)(address) };
            }
            for host in self
                .staging_pending
                .drain(..)
                .chain(self.staging_available.drain(..))
            {
                unsafe { (self.functions.pinned_free)(host.address) };
            }
            if !self.guard_module.is_null() {
                unsafe { (self.functions.module_unload)(self.guard_module) };
            }
            if !self.feedback_module.is_null() {
                unsafe { (self.functions.module_unload)(self.feedback_module) };
            }
            if !self.module.is_null() {
                unsafe { (self.functions.module_unload)(self.module) };
            }
            if !self.stream.is_null() {
                unsafe { (self.functions.stream_destroy)(self.stream) };
            }
            if context_pushed {
                let mut popped = ptr::null_mut();
                unsafe { (self.functions.context_pop)(&mut popped) };
                debug_assert_eq!(popped, self.context);
            }
            // Release exactly one primary-context reference *after* destroying
            // every per-session CUDA object and popping the current context.
            if !self.context.is_null() {
                unsafe { (self.functions.primary_release)(self.device) };
            }
        }
    }
    #[cfg(test)]
    mod batching_tests {
        use super::{descriptor_bytes, transfer_buckets};

        #[test]
        fn buckets_preserve_every_copy_and_bound_wasted_threads() {
            let lengths = [1024, 1, 257, 2, 513, 128, 1025, 4];
            let (order, buckets) = transfer_buckets(&lengths).unwrap();
            assert_eq!(order.len(), lengths.len());
            let mut unique = order.clone();
            unique.sort_unstable();
            assert_eq!(unique, (0..lengths.len()).collect::<Vec<_>>());
            let mut covered = 0;
            for (start, count, blocks) in buckets {
                assert_eq!(start as usize, covered);
                let slice = &order[start as usize..(start + count) as usize];
                let ceiling = lengths[slice[0]].next_power_of_two();
                assert!(
                    slice
                        .iter()
                        .all(|&index| lengths[index].next_power_of_two() == ceiling)
                );
                assert_eq!(
                    blocks as usize,
                    lengths[*slice.last().unwrap()].div_ceil(256)
                );
                assert!(count <= 65_535);
                covered += count as usize;
            }
            assert_eq!(covered, lengths.len());
        }

        #[test]
        fn descriptors_keep_full_pointer_bits_and_exact_lengths() {
            let entries = [
                [0xfeed_beef_dead_cafe, 0x1234_5678_ffff_0000, 4],
                [0, 0x8000_0000_0000_0000, 1025],
            ];
            let bytes = descriptor_bytes(&entries);
            assert_eq!(bytes.len(), entries.len() * 24);
            let reconstructed = bytes
                .chunks_exact(8)
                .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let expected = entries
                .iter()
                .flat_map(|e| e.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(reconstructed, expected);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::super::gpu::GpuError;

    pub(crate) struct CudaSession;

    impl CudaSession {
        pub fn new(_: &[u8]) -> Result<Self, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn allocate(&mut self, _: usize) -> Result<u64, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn require_allocation(&self, _: u64, _: usize) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_to_device(&mut self, _: u64, _: &[f32]) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_device_to_device(&mut self, _: u64, _: u64, _: usize) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }
        pub fn launch_feedback_copies(&mut self, _: &[(u64, u64, usize)]) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch_feedback_copies_device_timed(
            &mut self,
            _: &[(u64, u64, usize)],
        ) -> Result<std::time::Duration, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn validate_output_domains(
            &mut self,
            _: &[(u64, usize, bool)],
        ) -> Result<bool, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_f32_batch_to_host(
            &mut self,
            _: &[(u64, usize)],
        ) -> Result<Vec<Vec<f32>>, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_f32_batch_to_host_profiled(
            &mut self,
            _: &[(u64, usize)],
        ) -> Result<(Vec<Vec<f32>>, super::CudaD2hTiming), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_to_host(&mut self, _: &mut [f32], _: u64) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn copy_indices_to_host(&mut self, _: &mut [i32], _: u64) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn begin_graph(&mut self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn finish_graph(&mut self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch_graph(&self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn benchmark_graph(&self, _: usize, _: usize) -> Result<f32, GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn prepare_kernel(&mut self, _: &str) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch(
            &self,
            _: &str,
            _: u64,
            _: u64,
            _: u64,
            _: usize,
            _: u32,
        ) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch3(
            &self,
            _: &str,
            _: [u64; 3],
            _: u64,
            _: usize,
            _: u32,
        ) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch4(
            &self,
            _: &str,
            _: [u64; 4],
            _: u64,
            _: usize,
            _: u32,
        ) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn launch4_dual(
            &self,
            _: &str,
            _: [u64; 4],
            _: [u64; 2],
            _: usize,
            _: u32,
        ) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn synchronize(&self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }

        pub fn flush_input_copies(&mut self) -> Result<(), GpuError> {
            Err(GpuError::Driver(
                "CUDA Driver backend requires Linux".into(),
            ))
        }
    }
}

pub(super) use platform::CudaSession;
