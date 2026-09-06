const chartTitles = {
  '/1_distribution_analysis.png': 'Duration distribution',
  '/2_training_loss.png': 'Training and validation loss',
  '/3_actual_vs_predicted.png': 'Actual versus predicted duration',
  '/4_scatter_plot.png': 'Prediction scatter plot',
  '/5_residual_plot.png': 'Residual analysis',
};
const defaultCharts = Object.keys(chartTitles);

const number = (value) => {
  const parsed = Number.parseFloat(String(value).replace(/[^0-9.-]/g, ''));
  return Number.isFinite(parsed) ? parsed.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value;
};

function render(data) {
  const status = document.querySelector('#status');
  status.classList.toggle('ready', data.status === 'ready');
  status.innerHTML = `<span></span> ${data.status === 'ready' ? 'Results available' : 'Artifacts missing'}`;

  Object.entries(data.metrics || {}).forEach(([key, value]) => {
    const element = document.querySelector(`[data-metric="${key}"]`);
    if (element) element.textContent = number(value);
  });

  const config = document.querySelector('#configuration');
  Object.entries(data.configuration || {}).forEach(([key, value]) => {
    const item = document.createElement('div');
    item.className = 'config-item';
    item.innerHTML = `<span>${key}</span><strong>${Array.isArray(value) ? value.join(', ') : value}</strong>`;
    config.appendChild(item);
  });

  const predictions = document.querySelector('#predictions');
  predictions.replaceChildren();
  (data.predictions || []).slice(0, 20).forEach((row, index) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${index + 1}</td><td>${number(row.Actual)}</td><td>${number(row.Predicted)}</td><td>${number(row.Residual)}</td>`;
    predictions.appendChild(tr);
  });
  if (!predictions.children.length) predictions.innerHTML = '<tr><td colspan="4">No prediction artifact found.</td></tr>';

  renderCharts(data.charts || defaultCharts);
  document.querySelector('#summary').textContent = data.status === 'ready' ? 'Saved evaluation artifacts loaded from the latest local training run.' : 'Run the training pipeline locally and redeploy the generated artifacts.';
}

function renderCharts(sources) {
  const charts = document.querySelector('#charts');
  charts.replaceChildren();
  sources.forEach((src) => {
    const figure = document.createElement('figure');
    figure.className = 'chart';
    const image = document.createElement('img');
    image.src = new URL(src.replace(/^\/+/, ''), document.baseURI).href;
    image.alt = chartTitles[src] || 'Model chart';
    image.loading = 'lazy';
    image.addEventListener('error', () => {
      image.replaceWith(Object.assign(document.createElement('p'), {
        className: 'chart-error',
        textContent: 'Chart image is not available. Run the training pipeline to regenerate it.',
      }));
    }, { once: true });
    figure.append(image);
    const caption = document.createElement('figcaption');
    caption.textContent = chartTitles[src] || 'Model chart';
    figure.append(caption);
    charts.appendChild(figure);
  });
}

renderCharts(defaultCharts);

fetch('/api/results').then((response) => {
  if (!response.ok) throw new Error(`API returned ${response.status}`);
  return response.json();
}).then(render).catch(() => {
  document.querySelector('#status').innerHTML = '<span></span> API unavailable';
  document.querySelector('#summary').textContent = 'The results API could not be reached.';
});
