import {
    AutoProcessor,
    AutoModelForImageTextToText,
    load_image,
    TextStreamer,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';

// Global variables
let processor = null;
let model = null;
let isModelLoaded = false;
let currentDevice = 'cpu';

// DOM elements
const elements = {
    imageUrl: document.getElementById('image-url'),
    loadUrlBtn: document.getElementById('load-url-btn'),
    fileInput: document.getElementById('file-input'),
    uploadArea: document.getElementById('upload-area'),
    uploadBtn: document.getElementById('upload-btn'),
    previewSection: document.getElementById('preview-section'),
    previewImage: document.getElementById('preview-image'),
    customPrompt: document.getElementById('custom-prompt'),
    loadingSection: document.getElementById('loading-section'),
    loadingText: document.getElementById('loading-text'),
    progressFill: document.getElementById('progress-fill'),
    outputSection: document.getElementById('output-section'),
    outputContent: document.getElementById('output-content'),
    copyBtn: document.getElementById('copy-btn'),
    errorSection: document.getElementById('error-section'),
    errorMessage: document.getElementById('error-message'),
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    deviceRadios: document.querySelectorAll('input[name="device"]')
};

// Initialize model
async function initializeModel() {
    if (isModelLoaded && currentDevice === getSelectedDevice()) {
        return;
    }

    try {
        showLoading('Loading model...');
        currentDevice = getSelectedDevice();
        
        const model_id = "onnx-community/FastVLM-0.5B-ONNX";
        
        const modelOptions = {
            dtype: {
                embed_tokens: "fp16",
                vision_encoder: "q4",
                decoder_model_merged: "q4",
            }
        };

        if (currentDevice === 'webgpu') {
            modelOptions.device = 'webgpu';
        }

        updateLoadingText('Loading processor...');
        processor = await AutoProcessor.from_pretrained(model_id);
        
        updateLoadingText('Loading model...');
        model = await AutoModelForImageTextToText.from_pretrained(model_id, modelOptions);
        
        isModelLoaded = true;
        hideLoading();
    } catch (error) {
        console.error('Model initialization error:', error);
        showError('Failed to load model. Please try again.');
        hideLoading();
        throw error;
    }
}

// Generate caption for image
async function generateCaption(imageUrl) {
    try {
        hideError();
        showLoading('Processing image...');
        
        if (!isModelLoaded) {
            await initializeModel();
        }

        // Prepare prompt
        const customPromptText = elements.customPrompt.value.trim();
        const promptContent = customPromptText || "Describe this image in detail.";
        
        const messages = [
            {
                role: "user",
                content: `<image>${promptContent}`,
            },
        ];
        
        const prompt = processor.apply_chat_template(messages, {
            add_generation_prompt: true,
        });

        updateLoadingText('Loading image...');
        const image = await load_image(imageUrl);
        
        updateLoadingText('Processing inputs...');
        const inputs = await processor(image, prompt, {
            add_special_tokens: false,
        });

        updateLoadingText('Generating caption...');
        elements.outputContent.textContent = '';
        showOutput();

        const outputs = await model.generate({
            ...inputs,
            max_new_tokens: 512,
            do_sample: false,
            streamer: new TextStreamer(processor.tokenizer, {
                skip_prompt: true,
                skip_special_tokens: false,
                callback_function: (text) => {
                    elements.outputContent.textContent += text;
                },
            }),
        });

        const decoded = processor.batch_decode(
            outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
            { skip_special_tokens: true },
        );

        elements.outputContent.textContent = decoded[0];
        hideLoading();
    } catch (error) {
        console.error('Caption generation error:', error);
        showError('Failed to generate caption. Please check your image URL and try again.');
        hideLoading();
    }
}

// Helper functions
function getSelectedDevice() {
    const selected = document.querySelector('input[name="device"]:checked');
    return selected ? selected.value : 'cpu';
}

function showLoading(text) {
    elements.loadingSection.style.display = 'block';
    elements.loadingText.textContent = text;
    elements.progressFill.style.width = '50%';
}

function updateLoadingText(text) {
    elements.loadingText.textContent = text;
    const progress = {
        'Loading processor...': '30%',
        'Loading model...': '60%',
        'Loading image...': '70%',
        'Processing inputs...': '80%',
        'Generating caption...': '90%'
    };
    elements.progressFill.style.width = progress[text] || '50%';
}

function hideLoading() {
    elements.loadingSection.style.display = 'none';
    elements.progressFill.style.width = '0%';
}

function showOutput() {
    elements.outputSection.style.display = 'block';
}

function hideOutput() {
    elements.outputSection.style.display = 'none';
}

function showError(message) {
    elements.errorSection.style.display = 'block';
    elements.errorMessage.textContent = message;
}

function hideError() {
    elements.errorSection.style.display = 'none';
}

function showPreview(url) {
    elements.previewImage.src = url;
    elements.previewSection.style.display = 'block';
}

// Event listeners
elements.loadUrlBtn.addEventListener('click', async () => {
    const url = elements.imageUrl.value.trim();
    if (!url) {
        showError('Please enter a valid image URL');
        return;
    }
    showPreview(url);
    await generateCaption(url);
});

elements.uploadArea.addEventListener('click', () => {
    elements.fileInput.click();
});

elements.uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
});

elements.uploadArea.addEventListener('dragleave', () => {
    elements.uploadArea.classList.remove('dragover');
});

elements.uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

elements.fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

async function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    const url = URL.createObjectURL(file);
    showPreview(url);
    elements.uploadBtn.disabled = false;
    elements.uploadBtn.dataset.imageUrl = url;
}

elements.uploadBtn.addEventListener('click', async () => {
    const url = elements.uploadBtn.dataset.imageUrl;
    if (url) {
        await generateCaption(url);
    }
});

elements.copyBtn.addEventListener('click', () => {
    const text = elements.outputContent.textContent;
    navigator.clipboard.writeText(text).then(() => {
        elements.copyBtn.textContent = 'Copied!';
        setTimeout(() => {
            elements.copyBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                Copy Caption
            `;
        }, 2000);
    });
});

// Tab switching
elements.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.dataset.tab;
        
        elements.tabBtns.forEach(b => b.classList.remove('active'));
        elements.tabContents.forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`${targetTab}-tab`).classList.add('active');
    });
});

// Device selection
elements.deviceRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        if (isModelLoaded && currentDevice !== getSelectedDevice()) {
            isModelLoaded = false;
        }
    });
});