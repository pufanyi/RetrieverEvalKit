const backendSelect = document.getElementById('backendSelect');
    const topKInput = document.getElementById('topK');
    const searchForm = document.getElementById('searchForm');
    const textQueryGroup = document.getElementById('textQueryGroup');
    const captionQueryGroup = document.getElementById('captionQueryGroup');
    const imageQueryGroup = document.getElementById('imageQueryGroup');
    const textQueryInput = document.getElementById('textQuery');
    const captionIdInput = document.getElementById('captionId');
    const imageIdInput = document.getElementById('imageId');
    const statusInfo = document.getElementById('statusInfo');
    const resultsContainer = document.getElementById('results');
    const formError = document.getElementById('formError');
    const captionPreview = document.getElementById('captionPreview');

    function escapeHtml(unsafe) {
      if (unsafe === null || unsafe === undefined) return '';
      return String(unsafe)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    async function loadStatus() {
      try {
        const response = await fetch('/api/status');
        if (!response.ok) {
          throw new Error('Unable to load status');
        }
        const data = await response.json();
        renderStatus(data);
        populateBackends(data.methods);
        topKInput.value = data.default_top_k;
        topKInput.max = data.max_top_k;
        window.flickrDemoStatus = data;
      } catch (err) {
        statusInfo.textContent = err.message;
      }
    }

    function populateBackends(methods) {
      backendSelect.innerHTML = '';
      methods.forEach((method, index) => {
        const option = document.createElement('option');
        option.value = method.id;
        option.textContent = `${method.label} · ${method.metric}`;
        if (index === 0) {
          option.selected = true;
        }
        backendSelect.appendChild(option);
      });
    }

    function renderStatus(data) {
      const served = data.images_served ? 'served locally' : 'not served (files missing)';
      let html = `<div><strong>Embeddings:</strong> ${data.image_count.toLocaleString()} images · ${data.caption_count.toLocaleString()} captions</div>`;
      html += `<div><strong>Images:</strong> ${served}</div>`;
      if (data.sample_captions && data.sample_captions.length) {
        html += '<div style="margin-top:0.75rem"><strong>Sample caption IDs:</strong><ul style="margin:0.35rem 0 0 1.1rem; padding:0; list-style:disc;">';
        data.sample_captions.forEach((item) => {
          html += `<li><code>${escapeHtml(item.id)}</code> → <code>${escapeHtml(item.image_id ?? 'N/A')}</code> — ${escapeHtml(item.caption ?? '(no text)')}</li>`;
        });
        html += '</ul></div>';
      }
      if (Object.keys(data.method_errors || {}).length) {
        html += '<div class="error" style="margin-top:0.75rem">Unavailable backends:<ul style="margin:0.35rem 0 0 1.1rem; padding:0; list-style:disc;">';
        for (const [key, value] of Object.entries(data.method_errors)) {
          html += `<li><code>${escapeHtml(key)}</code> – ${escapeHtml(value)}</li>`;
        }
        html += '</ul></div>';
      }
      statusInfo.innerHTML = html;
    }

    function updateQueryMode() {
      const mode = document.querySelector('input[name="queryMode"]:checked').value;
      textQueryGroup.style.display = mode === 'text' ? 'block' : 'none';
      captionQueryGroup.style.display = mode === 'caption' ? 'block' : 'none';
      imageQueryGroup.style.display = mode === 'image' ? 'block' : 'none';
      formError.textContent = '';
    }

    async function previewCaption() {
      const id = captionIdInput.value.trim();
      captionPreview.style.display = 'none';
      captionPreview.textContent = '';
      if (!id) {
        return;
      }
      try {
        const response = await fetch(`/api/captions/${encodeURIComponent(id)}`);
        if (!response.ok) {
          throw new Error('Caption not found');
        }
        const data = await response.json();
        captionPreview.style.display = 'block';
        captionPreview.innerHTML = `<strong>${escapeHtml(data.id)}</strong> → ${escapeHtml(data.image_id ?? 'N/A')}<br/>${escapeHtml(data.caption ?? '(no caption text)')}`;
      } catch (err) {
        captionPreview.style.display = 'block';
        captionPreview.textContent = err.message;
      }
    }

    async function runSearch(event) {
      event.preventDefault();
      formError.textContent = '';
      const payload = {
        backend: backendSelect.value,
        top_k: Number(topKInput.value) || 9,
        query_mode: document.querySelector('input[name="queryMode"]:checked').value,
      };
      if (payload.query_mode === 'text') {
        payload.query = textQueryInput.value.trim();
        if (!payload.query) {
          formError.textContent = '请输入搜索文本';
          return;
        }
      } else if (payload.query_mode === 'caption') {
        payload.caption_id = captionIdInput.value.trim();
        if (!payload.caption_id) {
          formError.textContent = '请输入 caption ID';
          return;
        }
      } else {
        payload.image_id = imageIdInput.value.trim();
        if (!payload.image_id) {
          formError.textContent = '请输入 image ID';
          return;
        }
      }

      try {
        const response = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const message = await response.text();
          throw new Error(message || '搜索失败');
        }
        const data = await response.json();
        renderResults(data);
      } catch (err) {
        formError.textContent = err.message;
      }
    }

    function renderResults(data) {
      resultsContainer.innerHTML = '';
      if (!data.results.length) {
        const empty = document.createElement('p');
        empty.textContent = '未检索到结果。';
        resultsContainer.appendChild(empty);
        return;
      }
      data.results.forEach((hit) => {
        const card = document.createElement('article');
        card.className = 'result-card';

        if (hit.image_url) {
          const img = document.createElement('img');
          img.src = hit.image_url;
          img.alt = escapeHtml(hit.caption || hit.id);
          card.appendChild(img);
        } else {
          const placeholder = document.createElement('div');
          placeholder.style.display = 'flex';
          placeholder.style.alignItems = 'center';
          placeholder.style.justifyContent = 'center';
          placeholder.style.background = 'rgba(26,30,44,0.75)';
          placeholder.textContent = 'Image unavailable';
          card.appendChild(placeholder);
        }

        const meta = document.createElement('div');
        meta.className = 'result-meta';
        meta.innerHTML = `<h3>#${hit.rank} · ${escapeHtml(hit.id)}</h3>` +
          `<div class="metric">metric: ${escapeHtml(hit.metric)} · score: ${hit.score.toFixed(4)} · raw: ${hit.distance.toFixed(4)}</div>` +
          (hit.caption ? `<p style="margin:0.6rem 0 0 0;">${escapeHtml(hit.caption)}</p>` : '');

        if (hit.captions && hit.captions.length) {
          const list = document.createElement('ul');
          list.className = 'caption-list';
          hit.captions.forEach((entry) => {
            const item = document.createElement('li');
            item.innerHTML = `<code>${escapeHtml(entry.id)}</code> — ${escapeHtml(entry.caption ?? '(no text)')}`;
            list.appendChild(item);
          });
          meta.appendChild(list);
        }

        if (hit.image_path) {
          const pathInfo = document.createElement('div');
          pathInfo.style.marginTop = '0.65rem';
          pathInfo.style.fontSize = '0.8rem';
          pathInfo.style.opacity = '0.65';
          pathInfo.textContent = hit.image_path;
          meta.appendChild(pathInfo);
        }

        card.appendChild(meta);
        resultsContainer.appendChild(card);
      });
    }

    document.querySelectorAll('input[name="queryMode"]').forEach((input) => {
      input.addEventListener('change', updateQueryMode);
    });
    document.getElementById('previewCaption').addEventListener('click', previewCaption);
    searchForm.addEventListener('submit', runSearch);
    loadStatus();