import './style.css';
import { Scene3DManager } from './scene3d.js';
import { initializeLanguage, t } from './i18n.js';
import { ProcessCommand } from '../wailsjs/go/main/App.js';

let scene3DManager;
let animationRunning = false;
let lastTime = 0;
let frameCount = 0;
let fps = 0;
let currentGame = null;

// 游戏配置
const gameConfigs = {
    'web3d-sandbox': {
        title: 'games.sandbox.title',
        available: true
    },
    'space-explorer': {
        title: 'games.space.title',
        available: false
    },
    'cyber-city': {
        title: 'games.city.title',
        available: false
    },
    'quantum-lab': {
        title: 'games.quantum.title',
        available: false
    },
    'neural-network': {
        title: 'games.neural.title',
        available: false
    },
    'matrix-world': {
        title: 'games.matrix.title',
        available: false
    }
};

// 初始化应用
function initApp() {
    console.log('Initializing Tiny Games Hub...');
    
    // 初始化多语言
    initializeLanguage();
    
    // 显示主页
    showHomepage();
}

// 显示主页
function showHomepage() {
    document.getElementById('homepage').style.display = 'block';
    document.getElementById('game-page').classList.remove('active');
    
    // 清理3D场景
    if (scene3DManager) {
        scene3DManager.dispose();
        scene3DManager = null;
    }
    
    currentGame = null;
}

// 进入游戏
window.enterGame = function(gameId) {
    const gameConfig = gameConfigs[gameId];
    if (!gameConfig) {
        console.error('Unknown game:', gameId);
        return;
    }
    
    if (!gameConfig.available) {
        alert(t('status.coming_soon'));
        return;
    }
    
    currentGame = gameId;
    
    // 隐藏主页，显示游戏页面
    document.getElementById('homepage').style.display = 'none';
    document.getElementById('game-page').classList.add('active');
    
    // 更新游戏标题
    const titleElement = document.getElementById('current-game-title');
    if (titleElement) {
        titleElement.textContent = t(gameConfig.title);
    }
    
    // 初始化3D场景
    initGame();
};

// 退出游戏
window.exitGame = function() {
    showHomepage();
};

// 初始化游戏
function initGame() {
    console.log('Initializing game:', currentGame);
    
    // 初始化3D场景
    const container = document.getElementById('three-container');
    if (container) {
        scene3DManager = new Scene3DManager(container);
        console.log('3D Scene initialized');
        
        // 开始渲染循环
        animate();
    } else {
        console.error('Three.js container not found!');
    }
    
    // 设置键盘事件
    setupKeyboardEvents();
    
    // 更新统计信息
    updateStats();
}

// 动画循环
function animate(currentTime = 0) {
    requestAnimationFrame(animate);
    
    // 只在游戏页面激活时运行动画
    if (!document.getElementById('game-page').classList.contains('active')) {
        return;
    }
    
    // 计算FPS
    frameCount++;
    if (currentTime - lastTime >= 1000) {
        fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        frameCount = 0;
        lastTime = currentTime;
        updateFPS();
    }
    
    if (scene3DManager) {
        scene3DManager.animate(animationRunning);
    }
}

// 添加立方体
window.addCube = function() {
    if (scene3DManager) {
        scene3DManager.addCube();
        updateStats();
        console.log('Cube added');
    }
};

// 添加球体
window.addSphere = function() {
    if (scene3DManager) {
        scene3DManager.addSphere();
        updateStats();
        console.log('Sphere added');
    }
};

// 清空场景
window.clearScene = function() {
    if (scene3DManager) {
        scene3DManager.clearScene();
        updateStats();
        console.log('Scene cleared');
    }
};

// 切换动画
window.toggleAnimation = function() {
    animationRunning = !animationRunning;
    console.log('Animation:', animationRunning ? 'Started' : 'Stopped');
};

// 重置相机
window.resetCamera = function() {
    if (scene3DManager) {
        scene3DManager.resetCamera();
        console.log('Camera reset');
    }
};

// 发送消息到后端
window.sendMessage = async function() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) {
        alert(t('game.input_message'));
        return;
    }
    
    try {
        console.log('Sending message to backend:', message);
        const response = await ProcessCommand(message);
        console.log('Backend response:', response);
        
        // 更新后端响应显示
        const responseElement = document.getElementById('backend-response');
        if (responseElement) {
            responseElement.textContent = response;
        }
        
        // 处理后端响应
        handleBackendResponse(response, message);
        
        // 清空输入框
        input.value = '';
        
    } catch (error) {
        console.error('Error sending message to backend:', error);
        const responseElement = document.getElementById('backend-response');
        if (responseElement) {
            responseElement.textContent = '错误: ' + error.message;
        }
    }
};

// 处理后端响应
function handleBackendResponse(response, originalMessage) {
    const lowerResponse = response.toLowerCase();
    const lowerMessage = originalMessage.toLowerCase();
    
    // 根据响应内容执行相应的3D操作
    if (lowerResponse.includes('cube') || lowerMessage.includes('cube') || 
        lowerResponse.includes('立方体') || lowerMessage.includes('立方体')) {
        addCube();
    } else if (lowerResponse.includes('sphere') || lowerMessage.includes('sphere') || 
               lowerResponse.includes('球体') || lowerMessage.includes('球体')) {
        addSphere();
    } else if (lowerResponse.includes('clear') || lowerMessage.includes('clear') || 
               lowerResponse.includes('清空') || lowerMessage.includes('清空')) {
        clearScene();
    } else if (lowerResponse.includes('animation') || lowerMessage.includes('animation') || 
               lowerResponse.includes('动画') || lowerMessage.includes('动画')) {
        toggleAnimation();
    } else if (lowerResponse.includes('reset') || lowerMessage.includes('reset') || 
               lowerResponse.includes('重置') || lowerMessage.includes('重置')) {
        resetCamera();
    }
}

// 设置键盘事件
function setupKeyboardEvents() {
    document.addEventListener('keydown', (event) => {
        // 只在游戏页面激活时响应游戏快捷键
        if (!document.getElementById('game-page').classList.contains('active')) {
            return;
        }
        
        switch(event.key) {
            case '1':
                addCube();
                break;
            case '2':
                addSphere();
                break;
            case 'c':
            case 'C':
                clearScene();
                break;
            case ' ':
                event.preventDefault();
                toggleAnimation();
                break;
            case 'r':
            case 'R':
                resetCamera();
                break;
            case 'Escape':
                exitGame();
                break;
        }
    });
    
    // 回车键发送消息
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    }
}

// 更新统计信息
function updateStats() {
    const objectCountElement = document.getElementById('object-count');
    if (objectCountElement && scene3DManager) {
        objectCountElement.textContent = scene3DManager.getObjectCount();
    }
}

// 更新FPS显示
function updateFPS() {
    const fpsElement = document.getElementById('fps');
    if (fpsElement) {
        fpsElement.textContent = fps;
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initApp);

// 页面卸载时清理资源
window.addEventListener('beforeunload', () => {
    if (scene3DManager) {
        scene3DManager.dispose();
    }
});
