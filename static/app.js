/**
 * Z-Image 前端交互逻辑
 */

// ==================== 状态管理 ====================
const state = {
    models: [],
    defaultModel: '',
    selectedModel: '',
    isGenerating: false,
    currentImageBase64: null
};

// ==================== DOM 元素 ====================
const elements = {
    modelSelect: document.getElementById('model-select'),
    widthSlider: document.getElementById('width-slider'),
    widthValue: document.getElementById('width-value'),
    heightSlider: document.getElementById('height-slider'),
    heightValue: document.getElementById('height-value'),
    stepsSlider: document.getElementById('steps-slider'),
    stepsValue: document.getElementById('steps-value'),
    guidanceSlider: document.getElementById('guidance-slider'),
    guidanceValue: document.getElementById('guidance-value'),
    seedInput: document.getElementById('seed-input'),
    promptInput: document.getElementById('prompt-input'),
    negativePromptInput: document.getElementById('negative-prompt-input'),
    generateBtn: document.getElementById('generate-btn'),
    btnText: document.querySelector('.btn-text'),
    btnLoading: document.querySelector('.btn-loading'),
    resultContainer: document.getElementById('result-container'),
    resultImage: document.getElementById('result-image'),
    generationTime: document.getElementById('generation-time'),
    downloadBtn: document.getElementById('download-btn'),
    copyBtn: document.getElementById('copy-btn'),
    errorContainer: document.getElementById('error-container'),
    errorMessage: document.getElementById('error-message')
};

// ==================== API 调用 ====================
async function fetchModels() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) throw new Error('获取模型列表失败');
        
        const data = await response.json();
        state.models = data.models;
        state.defaultModel = data.default_model;
        
        renderModelSelect();
    } catch (error) {
        showError('无法加载模型列表: ' + error.message);
    }
}

async function fetchModelConfig(modelId) {
    try {
        const response = await fetch(`/api/config/${modelId}`);
        if (!response.ok) throw new Error('获取模型配置失败');
        
        const config = await response.json();
        applyModelConfig(config);
    } catch (error) {
        console.error('获取模型配置失败:', error);
    }
}

async function generateImage() {
    if (state.isGenerating) return;
    
    const prompt = elements.promptInput.value.trim();
    if (!prompt) {
        showError('请输入图像描述');
        elements.promptInput.focus();
        return;
    }
    
    setGenerating(true);
    hideError();
    hideResult();
    
    const requestBody = {
        prompt: prompt,
        model_id: state.selectedModel || state.defaultModel,
        negative_prompt: elements.negativePromptInput.value.trim(),
        width: parseInt(elements.widthSlider.value),
        height: parseInt(elements.heightSlider.value),
        steps: parseInt(elements.stepsSlider.value),
        guidance_scale: parseFloat(elements.guidanceSlider.value),
        save_to_file: false
    };
    
    const seed = elements.seedInput.value.trim();
    if (seed && !isNaN(parseInt(seed))) {
        requestBody.seed = parseInt(seed);
    }
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        if (result.success && result.image_base64) {
            showResult(result.image_base64, result.generation_time);
        } else {
            showError(result.error || '图像生成失败');
        }
    } catch (error) {
        showError('请求失败: ' + error.message);
    } finally {
        setGenerating(false);
    }
}

// ==================== UI 渲染 ====================
function renderModelSelect() {
    elements.modelSelect.innerHTML = '';
    
    state.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = `${model.name} (${model.type === 'local' ? '本地' : '远程'})`;
        
        if (model.id === state.defaultModel) {
            option.selected = true;
            state.selectedModel = model.id;
        }
        
        elements.modelSelect.appendChild(option);
    });
    
    // 加载默认模型配置
    if (state.selectedModel) {
        fetchModelConfig(state.selectedModel);
    }
}

function applyModelConfig(config) {
    if (config.default_params) {
        const params = config.default_params;
        
        if (params.width) {
            elements.widthSlider.value = params.width;
            elements.widthValue.textContent = params.width;
        }
        
        if (params.height) {
            elements.heightSlider.value = params.height;
            elements.heightValue.textContent = params.height;
        }
        
        if (params.steps) {
            elements.stepsSlider.value = params.steps;
            elements.stepsValue.textContent = params.steps;
        }
        
        if (params.guidance_scale) {
            elements.guidanceSlider.value = params.guidance_scale;
            elements.guidanceValue.textContent = params.guidance_scale;
        }
    }
}

function setGenerating(isGenerating) {
    state.isGenerating = isGenerating;
    elements.generateBtn.disabled = isGenerating;
    elements.btnText.style.display = isGenerating ? 'none' : 'inline';
    elements.btnLoading.style.display = isGenerating ? 'flex' : 'none';
}

function showResult(imageBase64, generationTime) {
    state.currentImageBase64 = imageBase64;
    elements.resultImage.src = `data:image/png;base64,${imageBase64}`;
    elements.generationTime.textContent = `生成耗时: ${generationTime.toFixed(2)}s`;
    elements.resultContainer.style.display = 'block';
    
    // 滚动到结果
    elements.resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideResult() {
    elements.resultContainer.style.display = 'none';
    state.currentImageBase64 = null;
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorContainer.style.display = 'flex';
}

function hideError() {
    elements.errorContainer.style.display = 'none';
}

// ==================== 下载和复制 ====================
function downloadImage() {
    if (!state.currentImageBase64) return;
    
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${state.currentImageBase64}`;
    link.download = `z-image-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function copyImageToClipboard() {
    if (!state.currentImageBase64) return;
    
    try {
        // 将 base64 转换为 blob
        const response = await fetch(`data:image/png;base64,${state.currentImageBase64}`);
        const blob = await response.blob();
        
        await navigator.clipboard.write([
            new ClipboardItem({ 'image/png': blob })
        ]);
        
        // 显示成功提示
        const originalText = elements.copyBtn.textContent;
        elements.copyBtn.textContent = '✅ 已复制';
        setTimeout(() => {
            elements.copyBtn.textContent = originalText;
        }, 2000);
    } catch (error) {
        showError('复制失败: ' + error.message);
    }
}

// ==================== 事件绑定 ====================
function bindEvents() {
    // 模型选择变化
    elements.modelSelect.addEventListener('change', (e) => {
        state.selectedModel = e.target.value;
        fetchModelConfig(e.target.value);
    });
    
    // 滑块实时更新
    elements.widthSlider.addEventListener('input', (e) => {
        elements.widthValue.textContent = e.target.value;
    });
    
    elements.heightSlider.addEventListener('input', (e) => {
        elements.heightValue.textContent = e.target.value;
    });
    
    elements.stepsSlider.addEventListener('input', (e) => {
        elements.stepsValue.textContent = e.target.value;
    });
    
    elements.guidanceSlider.addEventListener('input', (e) => {
        elements.guidanceValue.textContent = e.target.value;
    });
    
    // 生成按钮
    elements.generateBtn.addEventListener('click', generateImage);
    
    // 快捷键：Ctrl+Enter 生成
    elements.promptInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            generateImage();
        }
    });
    
    // 下载和复制
    elements.downloadBtn.addEventListener('click', downloadImage);
    elements.copyBtn.addEventListener('click', copyImageToClipboard);
}

// ==================== 初始化 ====================
function init() {
    bindEvents();
    fetchModels();
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);
