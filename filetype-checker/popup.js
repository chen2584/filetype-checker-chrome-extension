// FileType Checker - popup entry point
import * as ort from './lib/ort.bundle.min.mjs';

console.log('ONNX Runtime Web loaded successfully:', ort.env.versions?.common ?? 'unknown version');

// Module-level variable to store loaded config
let _configData = null;

/**
 * Load and parse the Magika label config from the bundled config.json.
 * Caches the result in a module-level variable for reuse.
 * @returns {Promise<Object>} The parsed config object with a labels array.
 */
async function loadConfig() {
  if (_configData) {
    return _configData;
  }
  const url = chrome.runtime.getURL('config.json');
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to load config.json: ' + response.statusText);
  }
  _configData = await response.json();
  return _configData;
}

/**
 * Get a label object by its model output index.
 * @param {number} index - The model output index.
 * @returns {Object|null} The label object matching the index, or null if not found.
 */
function getLabelByIndex(index) {
  if (!_configData || !_configData.labels) {
    return null;
  }
  return _configData.labels.find(label => label.index === index) || null;
}

/**
 * Look up the expected content type from a file extension.
 * Performs case-insensitive comparison against all labels' extensions arrays.
 * @param {string} extension - The file extension including the dot (e.g., ".pdf").
 * @returns {Object|null} The label object whose extensions contain the given extension, or null.
 */
function getExpectedTypeByExtension(extension) {
  if (!_configData || !_configData.labels || !extension) {
    return null;
  }
  const ext = extension.toLowerCase();
  return _configData.labels.find(label =>
    label.extensions.some(e => e.toLowerCase() === ext)
  ) || null;
}

// --- File Preprocessor ---

const REGION_SIZE = 1024;
const MIN_FILE_SIZE = 8;
const PADDING_TOKEN = 256;

/**
 * Preprocess a file's ArrayBuffer for Magika v3.3 model input.
 *
 * Extracts two 1024-byte regions (beginning and end) from the file,
 * pads with token 256 where needed, and returns raw byte values as
 * an Int32Array of length 2048 (1024 × 2).
 *
 * Layout: [beg_1024 | end_1024]
 *   - beg: first 1024 bytes, right-padded with 256 if file < 1024
 *   - end: last 1024 bytes, left-padded with 256 if file < 1024
 *
 * @param {ArrayBuffer} arrayBuffer - The raw file content.
 * @returns {Int32Array} An Int32Array of length 2048 ready for model input.
 * @throws {Error} If the file is empty or smaller than 8 bytes.
 */
function preprocessFile(arrayBuffer) {
  const fileSize = arrayBuffer.byteLength;

  if (fileSize === 0) {
    throw new Error('This file is empty and cannot be analyzed.');
  }

  if (fileSize < MIN_FILE_SIZE) {
    throw new Error('This file is too small to analyze. Please try a larger file.');
  }

  const fileBytes = new Uint8Array(arrayBuffer);
  const result = new Int32Array(REGION_SIZE * 2);

  // Fill entire buffer with padding token
  result.fill(PADDING_TOKEN);

  // Beginning: first min(1024, fileSize) bytes, right-padded with 256
  const begLength = Math.min(fileSize, REGION_SIZE);
  for (let i = 0; i < begLength; i++) {
    result[i] = fileBytes[i];
  }

  // End: last min(1024, fileSize) bytes, left-padded with 256
  const endOffset = REGION_SIZE;
  const endLength = Math.min(fileSize, REGION_SIZE);
  const endStart = fileSize - endLength;
  const leftPad = REGION_SIZE - endLength;
  for (let i = 0; i < endLength; i++) {
    result[endOffset + leftPad + i] = fileBytes[endStart + i];
  }

  return result;
}

// --- ONNX Model Loading and Inference ---

/**
 * Initialize the ONNX Runtime Web InferenceSession with the bundled Magika model.
 *
 * Configures the WASM backend paths to point to the extension's lib/ directory
 * and creates a session from the bundled model.onnx file.
 *
 * @returns {Promise<ort.InferenceSession>} The loaded ONNX inference session.
 * @throws {Error} If the model fails to load.
 */
async function initModel() {
  try {
    // Configure WASM backend for Chrome extension environment
    // The bundle build has the WASM binary inlined, so no wasmPaths needed.
    // Disable multi-threading (Chrome extension popups lack cross-origin isolation)
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    // Fetch the model as ArrayBuffer to avoid URL resolution issues
    const modelUrl = chrome.runtime.getURL('model.onnx');
    const modelResponse = await fetch(modelUrl);
    if (!modelResponse.ok) {
      throw new Error('model fetch failed');
    }
    const modelBuffer = await modelResponse.arrayBuffer();

    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm']
    });

    return session;
  } catch (error) {
    console.error('Model loading error:', error);
    if (error && error.message && error.message.includes('model')) {
      throw new Error('The AI model file appears to be damaged. Please reinstall the extension.');
    }
    throw new Error('Unable to load the AI model. Please try reinstalling the extension.');
  }
}

/**
 * Run inference on the ONNX model with the given preprocessed input tensor.
 *
 * @param {ort.InferenceSession} session - The loaded ONNX inference session.
 * @param {Int32Array} inputTensor - An Int32Array of length 2048 from preprocessFile().
 * @returns {Promise<Float32Array>} The raw output probabilities from the model.
 * @throws {Error} If inference fails.
 */
async function runInference(session, inputTensor) {
  try {
    const tensor = new ort.Tensor('int32', inputTensor, [1, 2048]);

    // Get the model's input name from the session
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: tensor };

    const results = await session.run(feeds);

    // Get the output tensor using the session's output name
    const outputName = session.outputNames[0];
    const output = results[outputName];

    return new Float32Array(output.data);
  } catch (error) {
    throw new Error('An error occurred during analysis. Please try again with a different file.');
  }
}

/**
 * Find the detected content type from model output probabilities.
 *
 * The Magika v3.3 model output is already probabilities (no softmax needed).
 * Finds the argmax index and maps it to a label using getLabelByIndex().
 *
 * @param {Float32Array} probabilities - Model output probabilities (shape [214]).
 * @returns {{ label: Object, confidence: number }} The detected label object and confidence score.
 */
function getDetectedType(probabilities) {
  // Model output is already probabilities — find argmax directly
  let maxIndex = 0;
  let maxProb = probabilities[0];
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }

  const label = getLabelByIndex(maxIndex);

  return {
    label: label,
    confidence: maxProb
  };
}

// --- Mismatch Comparison Logic ---

/**
 * Compare a file extension against the detected label's known extensions.
 *
 * Determines whether the file extension is found in the detected label's
 * extensions array (case-insensitive comparison).
 *
 * @param {string} fileExtension - The file extension including the dot (e.g., ".pdf"), or empty string.
 * @param {Object} detectedLabel - The detected label object from config with extensions array.
 * @returns {{ isMatch: boolean, status: string }}
 *   - isMatch: true if extension is in the detected label's extensions
 *   - status: 'match' | 'mismatch' | 'unknown_extension'
 */
function checkMismatch(fileExtension, detectedLabel) {
  // No extension or empty string → unknown
  if (!fileExtension) {
    return { isMatch: false, status: 'unknown_extension' };
  }

  const ext = fileExtension.toLowerCase();
  const knownExtensions = (detectedLabel.extensions || []).map(e => e.toLowerCase());

  if (knownExtensions.includes(ext)) {
    return { isMatch: true, status: 'match' };
  }

  return { isMatch: false, status: 'mismatch' };
}

/**
 * Build a complete FileAnalysisResult object from analysis data.
 *
 * @param {Object} params
 * @param {string} params.fileName - Original filename.
 * @param {string} params.fileExtension - Extracted extension (e.g., ".pdf"), or empty string.
 * @param {number} params.fileSize - File size in bytes.
 * @param {Object} params.detectedLabel - The detected label object from config.
 * @param {number} params.confidence - Model confidence score (0-1).
 * @returns {Object} A FileAnalysisResult object.
 */
function buildFileAnalysisResult({ fileName, fileExtension, fileSize, detectedLabel, confidence }) {
  const { isMatch, status } = checkMismatch(fileExtension, detectedLabel);

  return {
    fileName,
    fileExtension,
    fileSize,
    detectedType: {
      name: detectedLabel.name,
      description: detectedLabel.description,
      mimeType: detectedLabel.mime_type,
      group: detectedLabel.group,
      extensions: detectedLabel.extensions
    },
    confidence,
    isMatch,
    status
  };
}

// --- UI Wiring: Drag-and-Drop, File Upload, and Analysis Pipeline ---

// DOM element references
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results');
const resultCard = document.getElementById('result-card');
const resetBtn = document.getElementById('reset-btn');
const errorSection = document.getElementById('error');
const errorMessage = document.getElementById('error-message');

// App state
let appState = {
  phase: 'idle', // 'idle' | 'loading' | 'result' | 'error'
  modelLoaded: false,
  result: null,
  errorMessage: null
};

// Cached ONNX session
let _cachedSession = null;

/**
 * Extract the file extension from a filename (including the dot).
 * Returns empty string if no extension found.
 * @param {string} fileName
 * @returns {string} e.g. ".pdf" or ""
 */
function getFileExtension(fileName) {
  if (!fileName) return '';
  const lastDot = fileName.lastIndexOf('.');
  if (lastDot <= 0 || lastDot === fileName.length - 1) return '';
  return fileName.slice(lastDot).toLowerCase();
}

/**
 * Transition the UI to a given phase, showing/hiding sections accordingly.
 * @param {'idle' | 'loading' | 'result' | 'error'} phase
 */
function setPhase(phase) {
  appState.phase = phase;

  dropZone.classList.toggle('hidden', phase !== 'idle');
  loadingSection.classList.toggle('hidden', phase !== 'loading');
  resultsSection.classList.toggle('hidden', phase !== 'result');
  errorSection.classList.toggle('hidden', phase !== 'error');
}

/**
 * Format a file size in bytes to a human-readable string.
 * @param {number} bytes - File size in bytes.
 * @returns {string} Formatted size (e.g., "1.2 MB", "340 KB", "128 bytes").
 */
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' bytes';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Display an error message with retry guidance and switch to error phase.
 * @param {string} message
 */
function showError(message) {
  appState.errorMessage = message;
  errorMessage.innerHTML = '';

  const msgEl = document.createElement('p');
  msgEl.className = 'error-text';
  msgEl.textContent = message;
  errorMessage.appendChild(msgEl);

  const retryEl = document.createElement('p');
  retryEl.className = 'error-retry';
  retryEl.textContent = 'Please try again or check a different file.';
  errorMessage.appendChild(retryEl);

  const retryBtn = document.createElement('button');
  retryBtn.className = 'reset-btn';
  retryBtn.textContent = 'Try Again';
  retryBtn.addEventListener('click', () => {
    appState.result = null;
    appState.errorMessage = null;
    setPhase('idle');
  });
  errorMessage.appendChild(retryBtn);

  setPhase('error');
}

/**
 * Display the analysis result in a rich, color-coded result card.
 * Renders match (green), mismatch (red/orange), or unknown extension (blue) cards
 * with appropriate icons, details, and explanatory text.
 * @param {Object} result - A FileAnalysisResult object.
 */
function displayResult(result) {
  appState.result = result;

  // Clear previous classes and content
  resultCard.className = 'result-card';
  resultCard.innerHTML = '';

  const confidence = (result.confidence * 100).toFixed(1) + '%';
  const fileSize = formatFileSize(result.fileSize);

  if (result.status === 'match') {
    resultCard.classList.add('match');

    resultCard.innerHTML =
      '<div class="result-header">' +
        '<span class="result-icon">✅</span>' +
        '<span class="result-title">Content Matches Extension</span>' +
      '</div>' +
      '<div class="result-details">' +
        '<div class="result-row"><span class="result-label">File</span><span class="result-value">' + escapeHtml(result.fileName) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Size</span><span class="result-value">' + fileSize + '</span></div>' +
        '<div class="result-row"><span class="result-label">Extension</span><span class="result-value">' + escapeHtml(result.fileExtension) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Detected Type</span><span class="result-value">' + escapeHtml(result.detectedType.description) + '</span></div>' +
        '<div class="result-row"><span class="result-label">MIME Type</span><span class="result-value">' + escapeHtml(result.detectedType.mimeType) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Confidence</span><span class="result-value">' + confidence + '</span></div>' +
      '</div>';

  } else if (result.status === 'mismatch') {
    resultCard.classList.add('mismatch');

    const expectedType = getExpectedTypeByExtension(result.fileExtension);
    const expectedDesc = expectedType ? expectedType.description : 'Unknown';

    resultCard.innerHTML =
      '<div class="result-header">' +
        '<span class="result-icon">⚠️</span>' +
        '<span class="result-title">Content Does Not Match Extension</span>' +
      '</div>' +
      '<div class="result-details">' +
        '<div class="result-row"><span class="result-label">File</span><span class="result-value">' + escapeHtml(result.fileName) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Size</span><span class="result-value">' + fileSize + '</span></div>' +
        '<div class="result-row"><span class="result-label">Extension</span><span class="result-value">' + escapeHtml(result.fileExtension) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Expected Type</span><span class="result-value">' + escapeHtml(expectedDesc) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Detected Extension</span><span class="result-value">' + escapeHtml(result.detectedType.extensions.length > 0 ? result.detectedType.extensions.join(', ') : 'N/A') + '</span></div>' +
        '<div class="result-row"><span class="result-label">Detected Type</span><span class="result-value">' + escapeHtml(result.detectedType.description) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Confidence</span><span class="result-value">' + confidence + '</span></div>' +
      '</div>' +
      '<div class="result-explanation">The file content does not match its extension. This file may have been renamed or mislabeled.</div>';

  } else {
    // unknown_extension
    resultCard.classList.add('unknown');

    resultCard.innerHTML =
      '<div class="result-header">' +
        '<span class="result-icon">ℹ️</span>' +
        '<span class="result-title">Unknown File Extension</span>' +
      '</div>' +
      '<div class="result-details">' +
        '<div class="result-row"><span class="result-label">File</span><span class="result-value">' + escapeHtml(result.fileName) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Size</span><span class="result-value">' + fileSize + '</span></div>' +
        '<div class="result-row"><span class="result-label">Detected Type</span><span class="result-value">' + escapeHtml(result.detectedType.description) + '</span></div>' +
        '<div class="result-row"><span class="result-label">MIME Type</span><span class="result-value">' + escapeHtml(result.detectedType.mimeType) + '</span></div>' +
        '<div class="result-row"><span class="result-label">Confidence</span><span class="result-value">' + confidence + '</span></div>' +
      '</div>' +
      '<div class="result-explanation">File extension not recognized — detected content type shown.</div>';
  }

  setPhase('result');
}

/**
 * Escape HTML special characters to prevent XSS when inserting user-provided text.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/**
 * Analyze a single file through the full pipeline:
 * read → preprocess → infer → compare → display.
 * @param {File} file
 */
async function analyzeFile(file) {
  setPhase('loading');

  try {
    // Read file as ArrayBuffer
    const arrayBuffer = await file.arrayBuffer();

    // Preprocess
    const inputTensor = preprocessFile(arrayBuffer);

    // Load or reuse ONNX session
    if (!_cachedSession) {
      _cachedSession = await initModel();
      appState.modelLoaded = true;
    }

    // Run inference
    const probabilities = await runInference(_cachedSession, inputTensor);

    // Get detected type
    const { label: detectedLabel, confidence } = getDetectedType(probabilities);

    if (!detectedLabel) {
      showError('Unable to identify the file type. Please try a different file.');
      return;
    }

    // Extract file extension
    const fileExtension = getFileExtension(file.name);

    // Build result
    const result = buildFileAnalysisResult({
      fileName: file.name,
      fileExtension,
      fileSize: file.size,
      detectedLabel,
      confidence
    });

    // Display result
    displayResult(result);
  } catch (err) {
    showError(err.message || 'An error occurred during analysis. Please try again with a different file.');
  }
}

/**
 * Handle a file from either drag-and-drop or file input.
 * Ensures only a single file is processed at a time.
 * @param {File} file
 */
function handleFile(file) {
  if (!file || appState.phase === 'loading') return;
  analyzeFile(file);
}

// --- Drag-and-drop event handlers ---

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove('dragover');

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
});

// --- Click-to-browse handler ---

dropZone.addEventListener('click', () => {
  if (appState.phase === 'loading') return;
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    handleFile(fileInput.files[0]);
    fileInput.value = ''; // Reset so the same file can be re-selected
  }
});

// --- Reset button handler ---

resetBtn.addEventListener('click', () => {
  appState.result = null;
  appState.errorMessage = null;
  setPhase('idle');
});

// --- Initialization ---

// Pre-load config so it's ready before any file analysis
loadConfig().catch((err) => {
  console.error('Failed to pre-load config:', err);
});
