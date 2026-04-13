import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { pathToFileURL } from 'url';

// ── Load config.json directly for tests ──
const configPath = join(__dirname, '..', 'config.json');
const configData = JSON.parse(readFileSync(configPath, 'utf-8'));

// ── Re-implement the pure functions from popup.js for isolated testing ──
// These mirror the exact logic in popup.js but are decoupled from Chrome APIs.

const REGION_SIZE = 1024;
const MIN_FILE_SIZE = 8;
const PADDING_TOKEN = 256;

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

function getLabelByIndex(index) {
  if (!configData || !configData.labels) return null;
  return configData.labels.find(label => label.index === index) || null;
}

function getExpectedTypeByExtension(extension) {
  if (!configData || !configData.labels || !extension) return null;
  const ext = extension.toLowerCase();
  return configData.labels.find(label =>
    label.extensions.some(e => e.toLowerCase() === ext)
  ) || null;
}

function checkMismatch(fileExtension, detectedLabel) {
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

function getFileExtension(fileName) {
  if (!fileName) return '';
  const lastDot = fileName.lastIndexOf('.');
  if (lastDot <= 0 || lastDot === fileName.length - 1) return '';
  return fileName.slice(lastDot).toLowerCase();
}

function getDetectedType(probabilities) {
  // Model output is already probabilities — find argmax directly (no softmax)
  let maxIndex = 0;
  let maxProb = probabilities[0];
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }
  const label = getLabelByIndex(maxIndex);
  return { label, confidence: maxProb };
}


// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

// ── Preprocessor Tests ──
// Validates: Requirements 2.2, 2.5

describe('preprocessFile', () => {
  it('returns Int32Array of length 2048', () => {
    const buf = new ArrayBuffer(2048);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 2048; i++) bytes[i] = i % 256;
    const result = preprocessFile(buf);
    expect(result).toBeInstanceOf(Int32Array);
    expect(result.length).toBe(2048);
  });

  it('extracts correct beginning/end for a 2048-byte file', () => {
    const buf = new ArrayBuffer(2048);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 2048; i++) bytes[i] = i % 256;
    const result = preprocessFile(buf);

    // Beginning: first 1024 bytes
    for (let i = 0; i < 1024; i++) {
      expect(result[i]).toBe(bytes[i]);
    }
    // End: last 1024 bytes (starts at byte 1024)
    for (let i = 0; i < 1024; i++) {
      expect(result[1024 + i]).toBe(bytes[1024 + i]);
    }
  });

  it('pads correctly for a 100-byte file', () => {
    const buf = new ArrayBuffer(100);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 100; i++) bytes[i] = i + 1;
    const result = preprocessFile(buf);

    // Beginning: first 100 bytes filled, rest padded with 256
    for (let i = 0; i < 100; i++) {
      expect(result[i]).toBe(i + 1);
    }
    for (let i = 100; i < 1024; i++) {
      expect(result[i]).toBe(256);
    }

    // End: left-padded with 256, then all 100 bytes
    const endPad = 1024 - 100; // 924
    for (let i = 0; i < endPad; i++) {
      expect(result[1024 + i]).toBe(256);
    }
    for (let i = 0; i < 100; i++) {
      expect(result[1024 + endPad + i]).toBe(i + 1);
    }
  });

  it('handles exactly 1024-byte file (boundary case)', () => {
    const buf = new ArrayBuffer(1024);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 1024; i++) bytes[i] = i % 256;
    const result = preprocessFile(buf);

    // Beginning: all 1024 bytes
    for (let i = 0; i < 1024; i++) {
      expect(result[i]).toBe(bytes[i]);
    }
    // End: file >= REGION_SIZE, last 1024 bytes = all bytes (no padding)
    for (let i = 0; i < 1024; i++) {
      expect(result[1024 + i]).toBe(bytes[i]);
    }
  });

  it('uses raw integer byte values (no normalization)', () => {
    const buf = new ArrayBuffer(1024);
    const bytes = new Uint8Array(buf);
    // Set all bytes to 255 (max value)
    for (let i = 0; i < 1024; i++) bytes[i] = 255;
    const result = preprocessFile(buf);

    // Values should be raw integers, not normalized floats
    expect(result[0]).toBe(255);
    expect(result[1023]).toBe(255);
    // Padding should be 256
    // (no padding in this case since file == REGION_SIZE, but check end region)
    expect(result[1024]).toBe(255);
  });

  it('uses padding token 256 (not 0)', () => {
    const buf = new ArrayBuffer(8);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 8; i++) bytes[i] = i + 1;
    const result = preprocessFile(buf);

    // Beginning: 8 bytes then padding
    for (let i = 0; i < 8; i++) {
      expect(result[i]).toBe(i + 1);
    }
    expect(result[8]).toBe(256);
    expect(result[1023]).toBe(256);

    // End: left-padded with 256
    expect(result[1024]).toBe(256);
    expect(result[1024 + 1024 - 8 - 1]).toBe(256);
    // Last 8 bytes of end region
    for (let i = 0; i < 8; i++) {
      expect(result[1024 + (1024 - 8) + i]).toBe(i + 1);
    }
  });

  it('throws error for empty file (0 bytes)', () => {
    const buf = new ArrayBuffer(0);
    expect(() => preprocessFile(buf)).toThrow('empty');
  });

  it('throws error for file smaller than 8 bytes', () => {
    const buf = new ArrayBuffer(5);
    expect(() => preprocessFile(buf)).toThrow('too small');
  });

  it('accepts file of exactly 8 bytes (minimum)', () => {
    const buf = new ArrayBuffer(8);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 8; i++) bytes[i] = i;
    const result = preprocessFile(buf);
    expect(result.length).toBe(2048);
  });

  it('handles file between 1024 and 2048 bytes', () => {
    const buf = new ArrayBuffer(1500);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 1500; i++) bytes[i] = i % 256;
    const result = preprocessFile(buf);

    // Beginning: first 1024 bytes
    for (let i = 0; i < 1024; i++) {
      expect(result[i]).toBe(bytes[i]);
    }
    // End: last 1024 bytes, left-padded with 256
    // endLength = min(1024, 1500) = 1024, so no padding needed
    const endStart = 1500 - 1024; // 476
    for (let i = 0; i < 1024; i++) {
      expect(result[1024 + i]).toBe(bytes[endStart + i]);
    }
  });
});


// ── Label Mapping Tests ──
// Validates: Requirements 3.1

describe('getLabelByIndex', () => {
  it('returns PDF label for index 127', () => {
    const label = getLabelByIndex(127);
    expect(label).not.toBeNull();
    expect(label.name).toBe('pdf');
    expect(label.description).toBe('PDF document');
    expect(label.extensions).toContain('.pdf');
  });

  it('returns JPEG label for index 86', () => {
    const label = getLabelByIndex(86);
    expect(label).not.toBeNull();
    expect(label.name).toBe('jpeg');
    expect(label.extensions).toContain('.jpg');
    expect(label.extensions).toContain('.jpeg');
  });

  it('returns null for out-of-range index 999', () => {
    expect(getLabelByIndex(999)).toBeNull();
  });

  it('returns null for negative index', () => {
    expect(getLabelByIndex(-1)).toBeNull();
  });

  it('returns PNG label for index 133', () => {
    const label = getLabelByIndex(133);
    expect(label).not.toBeNull();
    expect(label.name).toBe('png');
    expect(label.extensions).toContain('.png');
  });

  it('has exactly 214 labels', () => {
    expect(configData.labels.length).toBe(214);
  });
});

describe('getExpectedTypeByExtension', () => {
  it('returns PDF label for .pdf', () => {
    const label = getExpectedTypeByExtension('.pdf');
    expect(label).not.toBeNull();
    expect(label.name).toBe('pdf');
  });

  it('handles case-insensitive match (.PDF)', () => {
    const label = getExpectedTypeByExtension('.PDF');
    expect(label).not.toBeNull();
    expect(label.name).toBe('pdf');
  });

  it('returns null for unknown extension .xyz', () => {
    expect(getExpectedTypeByExtension('.xyz')).toBeNull();
  });

  it('returns null for empty string', () => {
    expect(getExpectedTypeByExtension('')).toBeNull();
  });

  it('returns null for null', () => {
    expect(getExpectedTypeByExtension(null)).toBeNull();
  });

  it('returns JPEG label for .jpg', () => {
    const label = getExpectedTypeByExtension('.jpg');
    expect(label).not.toBeNull();
    expect(label.name).toBe('jpeg');
  });
});


// ── Mismatch Detection Tests ──
// Validates: Requirements 3.2, 3.3, 3.4

describe('checkMismatch', () => {
  const pdfLabel = { name: 'pdf', extensions: ['.pdf'], description: 'PDF document' };
  const jpegLabel = { name: 'jpeg', extensions: ['.jpg', '.jpeg'], description: 'JPEG image' };

  it('returns match when extension is in detected label extensions', () => {
    const result = checkMismatch('.pdf', pdfLabel);
    expect(result.isMatch).toBe(true);
    expect(result.status).toBe('match');
  });

  it('returns mismatch when extension is not in detected label extensions', () => {
    const result = checkMismatch('.txt', pdfLabel);
    expect(result.isMatch).toBe(false);
    expect(result.status).toBe('mismatch');
  });

  it('returns unknown_extension for empty string', () => {
    const result = checkMismatch('', pdfLabel);
    expect(result.isMatch).toBe(false);
    expect(result.status).toBe('unknown_extension');
  });

  it('returns unknown_extension for null', () => {
    const result = checkMismatch(null, pdfLabel);
    expect(result.isMatch).toBe(false);
    expect(result.status).toBe('unknown_extension');
  });

  it('returns unknown_extension for undefined', () => {
    const result = checkMismatch(undefined, pdfLabel);
    expect(result.isMatch).toBe(false);
    expect(result.status).toBe('unknown_extension');
  });

  it('handles case-insensitive extension matching', () => {
    const result = checkMismatch('.PDF', pdfLabel);
    expect(result.isMatch).toBe(true);
    expect(result.status).toBe('match');
  });

  it('matches any of multiple extensions', () => {
    expect(checkMismatch('.jpg', jpegLabel).isMatch).toBe(true);
    expect(checkMismatch('.jpeg', jpegLabel).isMatch).toBe(true);
  });
});


// ── getFileExtension Tests ──

describe('getFileExtension', () => {
  it('extracts .pdf from document.pdf', () => {
    expect(getFileExtension('document.pdf')).toBe('.pdf');
  });

  it('extracts .jpg from photo.jpg', () => {
    expect(getFileExtension('photo.jpg')).toBe('.jpg');
  });

  it('returns empty string for file with no extension', () => {
    expect(getFileExtension('README')).toBe('');
  });

  it('returns empty string for dotfile like .gitignore', () => {
    expect(getFileExtension('.gitignore')).toBe('');
  });

  it('returns empty string for null', () => {
    expect(getFileExtension(null)).toBe('');
  });

  it('returns empty string for empty string', () => {
    expect(getFileExtension('')).toBe('');
  });

  it('returns empty string when dot is last character', () => {
    expect(getFileExtension('file.')).toBe('');
  });

  it('handles multiple dots - returns last extension', () => {
    expect(getFileExtension('archive.tar.gz')).toBe('.gz');
  });

  it('lowercases the extension', () => {
    expect(getFileExtension('FILE.PDF')).toBe('.pdf');
  });
});

// ── getDetectedType Tests ──

describe('getDetectedType', () => {
  it('returns the label with highest probability', () => {
    // Create probabilities where index 127 (pdf) has the highest value
    const probs = new Float32Array(214);
    probs.fill(0.001);
    probs[127] = 0.95; // PDF
    const result = getDetectedType(probs);
    expect(result.label).not.toBeNull();
    expect(result.label.name).toBe('pdf');
    expect(result.confidence).toBeCloseTo(0.95, 3);
  });

  it('returns correct confidence for dominant probability', () => {
    const probs = new Float32Array(214);
    probs.fill(0.0);
    probs[86] = 0.99; // JPEG
    const result = getDetectedType(probs);
    expect(result.label.name).toBe('jpeg');
    expect(result.confidence).toBeCloseTo(0.99, 3);
  });

  it('does not apply softmax — uses raw probabilities', () => {
    // If softmax were applied, the result would be different
    const probs = new Float32Array(214);
    probs.fill(0.001);
    probs[133] = 0.8; // PNG
    const result = getDetectedType(probs);
    // Confidence should be exactly 0.8, not softmax-transformed
    expect(result.confidence).toBeCloseTo(0.8, 5);
    expect(result.label.name).toBe('png');
  });
});


// ── buildFileAnalysisResult Tests ──
// Validates: Requirements 3.2, 3.3, 3.4

describe('buildFileAnalysisResult', () => {
  it('builds a match result with all fields', () => {
    const pdfLabel = getLabelByIndex(127);
    const result = buildFileAnalysisResult({
      fileName: 'report.pdf',
      fileExtension: '.pdf',
      fileSize: 102400,
      detectedLabel: pdfLabel,
      confidence: 0.98
    });

    expect(result.fileName).toBe('report.pdf');
    expect(result.fileExtension).toBe('.pdf');
    expect(result.fileSize).toBe(102400);
    expect(result.detectedType.name).toBe('pdf');
    expect(result.detectedType.description).toBe('PDF document');
    expect(result.detectedType.mimeType).toBe('application/pdf');
    expect(result.detectedType.group).toBe('document');
    expect(result.detectedType.extensions).toContain('.pdf');
    expect(result.confidence).toBe(0.98);
    expect(result.isMatch).toBe(true);
    expect(result.status).toBe('match');
  });

  it('builds a mismatch result when extension does not match', () => {
    const pdfLabel = getLabelByIndex(127);
    const result = buildFileAnalysisResult({
      fileName: 'suspicious.txt',
      fileExtension: '.txt',
      fileSize: 50000,
      detectedLabel: pdfLabel,
      confidence: 0.95
    });

    expect(result.isMatch).toBe(false);
    expect(result.status).toBe('mismatch');
    expect(result.detectedType.name).toBe('pdf');
  });

  it('builds unknown_extension result for file with no extension', () => {
    const jpegLabel = getLabelByIndex(86);
    const result = buildFileAnalysisResult({
      fileName: 'noext',
      fileExtension: '',
      fileSize: 20000,
      detectedLabel: jpegLabel,
      confidence: 0.87
    });

    expect(result.isMatch).toBe(false);
    expect(result.status).toBe('unknown_extension');
    expect(result.detectedType.name).toBe('jpeg');
  });
});

// ── Full Pipeline Integration Test ──
// Validates: Requirements 2.2, 2.5, 3.1, 3.2, 3.3, 3.4, 4.2

describe('Full pipeline integration', () => {
  it('preprocesses a buffer, simulates inference, and builds correct result', () => {
    // 1. Create a known buffer (256 bytes, above minimum)
    const buf = new ArrayBuffer(256);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 256; i++) bytes[i] = i;

    // 2. Preprocess
    const tensor = preprocessFile(buf);
    expect(tensor).toBeInstanceOf(Int32Array);
    expect(tensor.length).toBe(2048);

    // 3. Simulate model output: probabilities with PDF (index 127) as highest
    const mockProbs = new Float32Array(214);
    mockProbs.fill(0.001);
    mockProbs[127] = 0.95; // PDF dominant

    // 4. Get detected type
    const { label: detectedLabel, confidence } = getDetectedType(mockProbs);
    expect(detectedLabel).not.toBeNull();
    expect(detectedLabel.name).toBe('pdf');

    // 5. Build result for a matching file
    const result = buildFileAnalysisResult({
      fileName: 'document.pdf',
      fileExtension: getFileExtension('document.pdf'),
      fileSize: 256,
      detectedLabel,
      confidence
    });

    expect(result.fileName).toBe('document.pdf');
    expect(result.fileExtension).toBe('.pdf');
    expect(result.fileSize).toBe(256);
    expect(result.detectedType.name).toBe('pdf');
    expect(result.confidence).toBeGreaterThan(0.9);
    expect(result.isMatch).toBe(true);
    expect(result.status).toBe('match');
  });

  it('detects mismatch in full pipeline when extension differs', () => {
    const buf = new ArrayBuffer(64);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 64; i++) bytes[i] = i;

    const tensor = preprocessFile(buf);
    expect(tensor.length).toBe(2048);

    // Simulate JPEG detection
    const mockProbs = new Float32Array(214);
    mockProbs.fill(0.001);
    mockProbs[86] = 0.95; // JPEG dominant

    const { label: detectedLabel, confidence } = getDetectedType(mockProbs);

    // File claims to be .txt but model says JPEG
    const result = buildFileAnalysisResult({
      fileName: 'fake.txt',
      fileExtension: getFileExtension('fake.txt'),
      fileSize: 64,
      detectedLabel,
      confidence
    });

    expect(result.isMatch).toBe(false);
    expect(result.status).toBe('mismatch');
    expect(result.detectedType.name).toBe('jpeg');
    expect(result.fileExtension).toBe('.txt');
  });

  it('handles file with no extension through full pipeline', () => {
    const buf = new ArrayBuffer(32);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < 32; i++) bytes[i] = 0xFF;

    const tensor = preprocessFile(buf);

    const mockProbs = new Float32Array(214);
    mockProbs.fill(0.001);
    mockProbs[133] = 0.90; // PNG

    const { label: detectedLabel, confidence } = getDetectedType(mockProbs);

    const result = buildFileAnalysisResult({
      fileName: 'mystery_file',
      fileExtension: getFileExtension('mystery_file'),
      fileSize: 32,
      detectedLabel,
      confidence
    });

    expect(result.status).toBe('unknown_extension');
    expect(result.fileExtension).toBe('');
    expect(result.detectedType.name).toBe('png');
  });
});
