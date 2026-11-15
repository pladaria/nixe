use super::*;
use crate::mme::{self, MacroProgram};
pub(super) struct Execution {
    context: Option<mme::Context>,
    jit: bool,
    proofs: Vec<Option<Arc<mme::Effects>>>,
    fifo: mme::Parameters,
}
impl Execution {
    pub fn new(jit: bool) -> Self {
        Self {
            context: None,
            jit,
            proofs: Vec::new(),
            fifo: Default::default(),
        }
    }
}
impl Preparation {
    pub fn install_macro(&mut self, program: MacroProgram) {
        let execution = self.macros.as_mut().unwrap();
        let context = execution
            .context
            .get_or_insert_with(|| mme::Context::new(program.code.clone(), execution.jit));
        if context.install(program.code) {
            execution.proofs.clear();
        }
        if program.start as usize >= execution.proofs.len() {
            execution
                .proofs
                .resize_with(program.start as usize + 1, || None);
        }
        execution.proofs[program.start as usize] = Some(program.effects);
    }
    pub fn run_macro(
        &mut self,
        source: Source,
        start: u32,
        parameters: &[u8],
        scheduler: &mut compute::Scheduler,
        host: &mut impl PreparationHost,
    ) -> Result<()> {
        // Taking the execution context allows callbacks to borrow preparation
        // directly. No executable allocator or invocation reference crosses threads.
        let mut execution = self.macros.take().expect("non-recursive macro execution");
        let result = (|| {
            let proof = execution
                .proofs
                .get(start as usize)
                .and_then(Option::as_ref)
                .ok_or("backend macro has no effect proof")?;
            if !proof.eligible || parameters.len() > 0x4000 {
                return Err("ineligible backend macro".into());
            }
            let mut fifo = execution.fifo.invocation(parameters);
            let mut callback = Host {
                preparation: self,
                scheduler,
                host,
                source,
                proof,
            };
            // Eligible macros have no data-RAM access. An empty slice retains
            // the execution-time bounds checks without duplicating channel RAM.
            execution.context.as_mut().unwrap().execute_start(
                start,
                &mut fifo,
                &mut [],
                &mut callback,
            )
        })();
        self.macros = Some(execution);
        result.map_err(|e| source.error(e))
    }
}
struct Host<'a, H> {
    preparation: &'a mut Preparation,
    scheduler: &'a mut compute::Scheduler,
    host: &'a mut H,
    source: Source,
    proof: &'a mme::Effects,
}
impl<H: PreparationHost> mme::Host for Host<'_, H> {
    fn state(&mut self, method: u32) -> Result<u32> {
        let regs = &self.preparation.engines[0].regs;
        if regs.method(method).is_none() {
            return Err(format!("MME STATE reads unknown method {method:#x}").into());
        }
        Ok(regs.get(method))
    }
    fn stopped(&self) -> bool {
        self.host.stopped()
    }
    #[cfg(target_arch = "x86_64")]
    fn bank(&self) -> (*const u32, *const u32, *const u64) {
        static LOCAL: [u64; 0xe00] = [0; 0xe00];
        let (bank, valid) = self.preparation.engines[0].regs.mme_bank();
        (bank, valid, LOCAL.as_ptr())
    }
    fn emit(&mut self, method: u32, mut value: u32) -> Result<mme::Emission> {
        if !self.proof.contains(method) {
            return Err(format!("backend MME effect proof violated before write: method={method:#x} value={value:#x}").into());
        }
        let regs = &self.preparation.engines[0].regs;
        let info = regs
            .method(method)
            .ok_or("backend MME emitted unknown method")?;
        if regs.get(0x124) == 3 {
            value = self.preparation.shadow.get(method as usize / 4);
        }
        let source = Source {
            method,
            argument: value,
            ..self.source
        };
        if info.ordinary() {
            return self
                .preparation
                .reconcile(source, self.host)
                .map(|()| mme::Emission::Complete);
        }
        if method == 0x560 {
            self.preparation.track(method, value);
            self.preparation.engines[0].regs.set(method, value);
            return Ok(mme::Emission::Dma);
        }
        self.preparation
            .execute(source, self.scheduler, self.host)
            .map(|()| mme::Emission::Complete)
    }
    fn dma(&mut self, method: u32, _dram: &mut [u32], fifo: &mut mme::Fifo<'_>) -> Result<()> {
        let count = self.preparation.engines[0].regs.get(method) as usize;
        if count > mme::DRAM_WORDS {
            return Err("MME DMA read exceeds FIFO".into());
        }
        if count == 0 {
            return Ok(());
        }
        self.preparation.flush(self.host)?;
        let address = self.preparation.engines[0].regs.address(0x550, 0x554);
        fifo.read_append(count, |destination| {
            self.host.read_into(address, destination)
        })
    }
}
