// AudioGen — Minimalist Speech Synthesis Client

(function() {
  'use strict';

  const state = {
    step: 1,
    language: 'en',
    voice: null,
    script: '',
    audioBlobUrl: null
  };

  const LANG_LABELS = {
    en: 'English (EN)',
    hi: 'Hindi (HI)',
    pa: 'Punjabi (PA)'
  };

  // Elements
  const stepIndicator = document.getElementById('step-indicator');
  const panels = {
    1: document.getElementById('step-1'),
    2: document.getElementById('step-2'),
    3: document.getElementById('step-3'),
    4: document.getElementById('step-4')
  };

  const step2LangBadge = document.getElementById('step2-lang-badge');
  const step3VoiceBadge = document.getElementById('step3-voice-badge');
  const voiceGrid = document.getElementById('voice-grid');
  const openVoiceModalBtn = document.getElementById('open-voice-modal-btn');
  const voiceModal = document.getElementById('voice-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const voiceForm = document.getElementById('voice-form');

  const scriptInput = document.getElementById('script-input');
  const charCounter = document.getElementById('char-counter');
  const btnGenerate = document.getElementById('btn-generate');

  const progressState = document.getElementById('progress-state');
  const outputState = document.getElementById('output-state');
  const errorState = document.getElementById('error-state');
  const errorMessage = document.getElementById('error-message');
  const audioPlayer = document.getElementById('audio-player');
  const btnDownload = document.getElementById('btn-download');
  const btnReset = document.getElementById('btn-reset');
  const btnRetry = document.getElementById('btn-retry');

  // Navigation
  function goToStep(stepNum) {
    state.step = stepNum;
    stepIndicator.textContent = `Step ${stepNum} of 4`;

    Object.keys(panels).forEach(key => {
      const p = panels[key];
      if (parseInt(key, 10) === stepNum) {
        p.classList.remove('hidden');
        p.classList.add('active');
      } else {
        p.classList.add('hidden');
        p.classList.remove('active');
      }
    });
  }

  // Step 1: Language Selection
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const lang = btn.getAttribute('data-lang');
      selectLanguage(lang);
    });
  });

  async function selectLanguage(lang) {
    state.language = lang;
    step2LangBadge.textContent = LANG_LABELS[lang] || lang.toUpperCase();
    goToStep(2);
    await loadVoices(lang);
  }

  // Step 2: Voice Selection
  async function loadVoices(lang) {
    voiceGrid.innerHTML = '<p class="step-desc" style="margin-bottom:0;">Loading voice profiles...</p>';
    try {
      const res = await fetch(`/api/voices?language=${encodeURIComponent(lang)}`);
      if (!res.ok) throw new Error('Failed to fetch voices');
      const voices = await res.json();
      renderVoices(voices);
    } catch (err) {
      voiceGrid.innerHTML = `<p class="step-desc error-title">Error loading voices: ${err.message}</p>`;
    }
  }

  function renderVoices(voices) {
    voiceGrid.innerHTML = '';
    if (!voices || voices.length === 0) {
      voiceGrid.innerHTML = '<p class="step-desc" style="margin-bottom:0;">No voices found for this language. Create one below.</p>';
      return;
    }

    voices.forEach(voice => {
      const card = document.createElement('div');
      card.className = 'voice-card';
      const voiceId = voice.id || voice.speaker_ref_name;
      const displayName = voice.name || voiceId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

      const langTagsHtml = (voice.language || []).map(l => `<span class="voice-tag">${l.toUpperCase()}</span>`).join('');

      card.innerHTML = `
        <div class="voice-card-header">
          <div class="voice-card-title">${displayName}</div>
          <div class="voice-card-tags">${langTagsHtml}</div>
        </div>
        <div class="voice-card-desc">${voice.description || 'Reference speech synthesis profile.'}</div>
      `;

      card.addEventListener('click', () => {
        selectVoice(voiceId, displayName);
      });

      voiceGrid.appendChild(card);
    });
  }

  function selectVoice(voiceId, displayName) {
    state.voice = voiceId;
    step3VoiceBadge.textContent = displayName;
    goToStep(3);
  }

  // Back buttons
  document.querySelectorAll('.nav-back-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = parseInt(btn.getAttribute('data-target'), 10);
      goToStep(target);
    });
  });

  // Voice Modal
  openVoiceModalBtn.addEventListener('click', () => {
    document.getElementById('new-voice-lang').value = state.language;
    voiceModal.classList.remove('hidden');
  });

  btnCloseModal.addEventListener('click', () => {
    voiceModal.classList.add('hidden');
  });

  voiceModal.addEventListener('click', (e) => {
    if (e.target === voiceModal) {
      voiceModal.classList.add('hidden');
    }
  });

  voiceForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const id = document.getElementById('new-voice-id').value.trim();
    const lang = document.getElementById('new-voice-lang').value;
    const desc = document.getElementById('new-voice-desc').value.trim();
    const samplePath = document.getElementById('new-voice-sample').value;
    const refText = document.getElementById('new-voice-ref-text').value.trim();

    if (!id || !refText) return;

    try {
      const payload = {
        id: id,
        name: id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        language: [lang],
        description: desc || `Custom voice ${id}`,
        path: samplePath,
        ref_text: refText
      };

      const res = await fetch('/api/voices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorData = await res.json();
        alert(`Failed to save voice: ${errorData.detail || 'Unknown error'}`);
        return;
      }

      voiceModal.classList.add('hidden');
      voiceForm.reset();

      if (state.language !== lang) {
        await selectLanguage(lang);
      } else {
        await loadVoices(lang);
      }
      selectVoice(id, payload.name);
    } catch (err) {
      alert(`Network error saving voice profile: ${err.message}`);
    }
  });

  // Step 3: Script Input & Counter
  scriptInput.addEventListener('input', () => {
    state.script = scriptInput.value;
    const len = state.script.length;
    charCounter.textContent = `${len} ${len === 1 ? 'character' : 'characters'}`;
    btnGenerate.disabled = state.script.trim().length === 0;
  });

  btnGenerate.addEventListener('click', () => {
    if (btnGenerate.disabled) return;
    startGeneration();
  });

  // Step 4: Progress, Synthesis & Output
  async function startGeneration() {
    goToStep(4);
    progressState.classList.remove('hidden');
    outputState.classList.add('hidden');
    errorState.classList.add('hidden');

    try {
      const payload = {
        text: state.script.trim(),
        language: state.language,
        speaker_ref_name: state.voice
      };

      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        let errDetail = 'Synthesis failed. Please try again.';
        try {
          const errData = await res.json();
          if (errData && errData.detail) {
            errDetail = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch (_) {}
        throw new Error(errDetail);
      }

      const audioBlob = await res.blob();
      if (state.audioBlobUrl) {
        URL.revokeObjectURL(state.audioBlobUrl);
      }
      state.audioBlobUrl = URL.createObjectURL(audioBlob);

      audioPlayer.src = state.audioBlobUrl;
      btnDownload.href = state.audioBlobUrl;
      btnDownload.download = `audiogen_${state.language}_${Date.now()}.wav`;

      progressState.classList.add('hidden');
      outputState.classList.remove('hidden');
    } catch (err) {
      progressState.classList.add('hidden');
      errorMessage.textContent = err.message;
      errorState.classList.remove('hidden');
    }
  }

  btnRetry.addEventListener('click', () => {
    goToStep(3);
  });

  btnReset.addEventListener('click', () => {
    if (state.audioBlobUrl) {
      URL.revokeObjectURL(state.audioBlobUrl);
      state.audioBlobUrl = null;
    }
    audioPlayer.src = '';
    btnDownload.href = '#';
    state.script = '';
    state.voice = null;
    scriptInput.value = '';
    charCounter.textContent = '0 characters';
    btnGenerate.disabled = true;
    goToStep(1);
  });

})();
