import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, ArrowRight, CheckCircle2 } from 'lucide-react';
import toast from 'react-hot-toast';

export default function UploadView({ state }) {
  const {
    isLoading, isUploadingResume,
    isDragActive, setIsDragActive,
    file, setFile,
    status, resumeUploaded,
    role, setRole,
    analyzeRoleAndRunGap, uploadResumeStep,
  } = state;

  const statusIsError = /failed|error|missing|expired|not found|unsupported|does not appear/i.test(status);

  function handleDrop(e) {
    e.preventDefault();
    setIsDragActive(false);
    const dropped = e.dataTransfer?.files?.[0];
    if (dropped) {
      if (dropped.type !== 'application/pdf') {
        toast.error('Only PDF files are supported');
        return;
      }
      if (dropped.size > 5 * 1024 * 1024) {
        toast.error('File too large — max 5 MB');
        return;
      }
      setFile(dropped);
      toast.success(`Selected: ${dropped.name}`);
    }
  }

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      toast.success(`Selected: ${f.name}`);
    }
  }

  return (
    <motion.div
      className="stage"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="stage-card">
        {/* Step indicator */}
        <div className="step-indicator">
          <div className={`step-dot ${resumeUploaded ? 'step-done' : 'step-active'}`}>
            {resumeUploaded ? <CheckCircle2 size={14} /> : '1'}
          </div>
          <div className="step-line" />
          <div className={`step-dot ${resumeUploaded ? 'step-active' : 'step-pending'}`}>
            2
          </div>
        </div>

        <AnimatePresence mode="wait">
          {/* ── Step 1: Upload ── */}
          {!resumeUploaded && (
            <motion.div
              key="step-upload"
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 16 }}
              transition={{ duration: 0.35 }}
            >
              <div className="kicker">
                <span className="kicker-dot" />
                Step 1 — Upload Resume
              </div>

              <h1 className="stage-title">
                Connect your expertise.
              </h1>
              <p className="stage-sub">
                Upload your resume and our AI will instantly extract your professional profile,
                skills, and experience.
              </p>

              {/* Dropzone */}
              <div
                className={`dropzone${isDragActive ? ' drag-over' : ''}${file ? ' has-file' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
                onDragLeave={() => setIsDragActive(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('resume-file-input').click()}
              >
                <input
                  id="resume-file-input"
                  className="file-input-hidden"
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                />

                <motion.div
                  className="dropzone-icon"
                  animate={isDragActive
                    ? { scale: 1.25, rotate: 10, color: 'var(--accent)' }
                    : file
                      ? { scale: 1, rotate: 0, color: 'var(--success)' }
                      : { scale: 1, rotate: 0, color: 'var(--txt-3)' }
                  }
                >
                  {file ? <CheckCircle2 size={40} strokeWidth={1.5} /> : <Upload size={40} strokeWidth={1.5} />}
                </motion.div>

                <p className="dropzone-label">
                  {isDragActive ? 'Release to upload' : file ? file.name : 'Drop your resume here'}
                </p>
                <p className="dropzone-sub">
                  {file ? `${(file.size / 1024).toFixed(0)} KB • PDF` : 'PDF format · Max 5 MB'}
                </p>

                {!isUploadingResume && !file && (
                  <label
                    htmlFor="resume-file-input"
                    className="btn-ghost"
                    style={{ fontSize: '0.82rem', padding: '8px 18px', cursor: 'pointer' }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <FileText size={14} /> Browse Files
                  </label>
                )}

                {isUploadingResume && (
                  <div style={{ textAlign: 'center', marginTop: 12 }}>
                    <span className="upload-ring" />
                    <p style={{ marginTop: 10, fontSize: '0.82rem', color: 'var(--txt-3)' }}>
                      Parsing document…
                    </p>
                  </div>
                )}
              </div>

              <motion.button
                className="btn-primary"
                style={{ width: '100%', padding: '16px' }}
                onClick={uploadResumeStep}
                disabled={isLoading || !file}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
              >
                {isLoading ? (
                  <><span className="upload-ring" style={{ width: 16, height: 16 }} /> Processing…</>
                ) : (
                  <>Upload & Extract <ArrowRight size={16} /></>
                )}
              </motion.button>
            </motion.div>
          )}

          {/* ── Step 2: Role Input ── */}
          {resumeUploaded && (
            <motion.div
              key="step-role"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.35 }}
            >
              <div className="kicker">
                <span className="kicker-dot" />
                Step 2 — Define Target
              </div>

              <h1 className="stage-title">
                Set your target role.
              </h1>
              <p className="stage-sub">
                Enter the job title you're aiming for and we'll benchmark your skills
                against real-world requirements.
              </p>

              <div className="input-group">
                <label className="input-label" htmlFor="role-input">
                  Target Job Title
                </label>
                <input
                  id="role-input"
                  className="noir-input"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !isLoading) analyzeRoleAndRunGap(); }}
                  placeholder="e.g. Senior Backend Engineer, AI Engineer…"
                />
              </div>

              <div className="role-suggestions">
                {['Frontend Developer', 'AI Engineer', 'Data Scientist', 'Product Manager'].map(r => (
                  <button
                    key={r}
                    className="role-chip"
                    onClick={() => setRole(r)}
                    type="button"
                  >
                    {r}
                  </button>
                ))}
              </div>

              <motion.button
                className="btn-primary"
                style={{ width: '100%', padding: '16px', marginTop: 12 }}
                onClick={analyzeRoleAndRunGap}
                disabled={isLoading || !role.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
              >
                {isLoading ? (
                  <><span className="upload-ring" style={{ width: 16, height: 16 }} /> Analyzing…</>
                ) : (
                  <>Start Benchmarking <ArrowRight size={16} /></>
                )}
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Status toast (fallback) */}
        {status && (
          <p
            className={`status-bar ${statusIsError ? 'status-error' : 'status-info'}`}
            style={{ marginTop: 20 }}
            role={statusIsError ? 'alert' : 'status'}
          >
            {status}
          </p>
        )}
      </div>
    </motion.div>
  );
}
