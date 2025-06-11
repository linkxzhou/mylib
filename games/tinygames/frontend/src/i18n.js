// 多语言支持模块
const translations = {
    zh: {
        title: "迷你游戏中心",
        hero: {
            title: "迷你游戏中心",
            subtitle: "探索未来世界的3D游戏体验"
        },
        games: {
            sandbox: {
                title: "3D沙盒世界",
                description: "在这个3D世界中创建和操控各种几何体，体验实时物理引擎和动画效果。"
            },
            space: {
                title: "太空探索者",
                description: "驾驶宇宙飞船探索神秘的太空，发现新的星球和文明。"
            },
            city: {
                title: "赛博都市",
                description: "在霓虹闪烁的未来都市中冒险，体验赛博朋克风格的3D世界。"
            },
            quantum: {
                title: "量子实验室",
                description: "在量子物理的奇妙世界中进行实验，观察粒子的神奇行为。"
            },
            neural: {
                title: "神经网络",
                description: "可视化人工智能的学习过程，构建和训练你自己的神经网络。"
            },
            matrix: {
                title: "矩阵世界",
                description: "进入数字矩阵，体验虚拟现实与真实世界的边界模糊。"
            }
        },
        status: {
            available: "可游玩",
            coming_soon: "即将推出"
        },
        footer: {
            copyright: "© 2024 Tiny Games Hub. 探索无限可能的游戏世界。"
        },
        game: {
            back: "← 返回主页",
            controls: "游戏控制",
            scene_control: "场景控制:",
            add_cube: "添加立方体",
            add_sphere: "添加球体",
            clear_scene: "清空场景",
            animation_control: "动画控制:",
            toggle_animation: "开始/停止动画",
            reset_camera: "重置相机",
            backend_communication: "与后端通信:",
            input_message: "输入消息",
            send_message: "发送到后端",
            object_count: "对象数量:",
            backend_response: "后端响应:"
        }
    },
    en: {
        title: "TINY GAMES HUB",
        hero: {
            title: "Tiny Games Hub",
            subtitle: "Explore 3D gaming experiences in the future world"
        },
        games: {
            sandbox: {
                title: "3D Sandbox World",
                description: "Create and manipulate various geometric objects in this 3D world, experience real-time physics engine and animation effects."
            },
            space: {
                title: "Space Explorer",
                description: "Pilot spaceships to explore mysterious space, discover new planets and civilizations."
            },
            city: {
                title: "Cyber City",
                description: "Adventure in neon-lit future cities, experience cyberpunk-style 3D worlds."
            },
            quantum: {
                title: "Quantum Lab",
                description: "Conduct experiments in the wonderful world of quantum physics, observe the magical behavior of particles."
            },
            neural: {
                title: "Neural Network",
                description: "Visualize the learning process of artificial intelligence, build and train your own neural networks."
            },
            matrix: {
                title: "Matrix World",
                description: "Enter the digital matrix, experience the blurred boundaries between virtual reality and the real world."
            }
        },
        status: {
            available: "Available",
            coming_soon: "Coming Soon"
        },
        footer: {
            copyright: "© 2024 Tiny Games Hub. Explore infinite possibilities in gaming worlds."
        },
        game: {
            back: "← Back to Home",
            controls: "Game Controls",
            scene_control: "Scene Control:",
            add_cube: "Add Cube",
            add_sphere: "Add Sphere",
            clear_scene: "Clear Scene",
            animation_control: "Animation Control:",
            toggle_animation: "Start/Stop Animation",
            reset_camera: "Reset Camera",
            backend_communication: "Backend Communication:",
            input_message: "Enter message",
            send_message: "Send to Backend",
            object_count: "Object Count:",
            backend_response: "Backend Response:"
        }
    },
    ja: {
        title: "タイニーゲームハブ",
        hero: {
            title: "タイニーゲームハブ",
            subtitle: "未来世界の3Dゲーム体験を探索"
        },
        games: {
            sandbox: {
                title: "3Dサンドボックス世界",
                description: "この3D世界で様々な幾何学的オブジェクトを作成・操作し、リアルタイム物理エンジンとアニメーション効果を体験。"
            },
            space: {
                title: "宇宙探検家",
                description: "宇宙船を操縦して神秘的な宇宙を探索し、新しい惑星と文明を発見。"
            },
            city: {
                title: "サイバーシティ",
                description: "ネオンが輝く未来都市で冒険し、サイバーパンクスタイルの3D世界を体験。"
            },
            quantum: {
                title: "量子実験室",
                description: "量子物理学の素晴らしい世界で実験を行い、粒子の魔法的な振る舞いを観察。"
            },
            neural: {
                title: "ニューラルネットワーク",
                description: "人工知能の学習プロセスを可視化し、独自のニューラルネットワークを構築・訓練。"
            },
            matrix: {
                title: "マトリックス世界",
                description: "デジタルマトリックスに入り、仮想現実と現実世界の境界の曖昧さを体験。"
            }
        },
        status: {
            available: "プレイ可能",
            coming_soon: "近日公開"
        },
        footer: {
            copyright: "© 2024 Tiny Games Hub. ゲーム世界の無限の可能性を探索。"
        },
        game: {
            back: "← ホームに戻る",
            controls: "ゲームコントロール",
            scene_control: "シーンコントロール:",
            add_cube: "キューブを追加",
            add_sphere: "球体を追加",
            clear_scene: "シーンをクリア",
            animation_control: "アニメーションコントロール:",
            toggle_animation: "アニメーション開始/停止",
            reset_camera: "カメラをリセット",
            backend_communication: "バックエンド通信:",
            input_message: "メッセージを入力",
            send_message: "バックエンドに送信",
            object_count: "オブジェクト数:",
            backend_response: "バックエンド応答:"
        }
    }
};

let currentLanguage = 'zh';

// 切换语言
function switchLanguage(lang) {
    currentLanguage = lang;
    updateLanguageButtons();
    updatePageContent();
    localStorage.setItem('preferredLanguage', lang);
}

// 更新语言按钮状态
function updateLanguageButtons() {
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const langButtons = {
        'zh': 0,
        'en': 1,
        'ja': 2
    };
    
    const buttons = document.querySelectorAll('.lang-btn');
    if (buttons[langButtons[currentLanguage]]) {
        buttons[langButtons[currentLanguage]].classList.add('active');
    }
}

// 更新页面内容
function updatePageContent() {
    const elements = document.querySelectorAll('[data-i18n]');
    elements.forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = getNestedTranslation(translations[currentLanguage], key);
        if (translation) {
            element.textContent = translation;
        }
    });
    
    // 更新placeholder
    const placeholderElements = document.querySelectorAll('[data-i18n-placeholder]');
    placeholderElements.forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        const translation = getNestedTranslation(translations[currentLanguage], key);
        if (translation) {
            element.placeholder = translation;
        }
    });
}

// 获取嵌套翻译
function getNestedTranslation(obj, key) {
    return key.split('.').reduce((o, k) => o && o[k], obj);
}

// 初始化语言
function initializeLanguage() {
    const savedLanguage = localStorage.getItem('preferredLanguage');
    if (savedLanguage && translations[savedLanguage]) {
        currentLanguage = savedLanguage;
    }
    updateLanguageButtons();
    updatePageContent();
}

// 获取当前语言的翻译
function t(key) {
    return getNestedTranslation(translations[currentLanguage], key) || key;
}

// 导出函数
window.switchLanguage = switchLanguage;
window.initializeLanguage = initializeLanguage;
window.t = t;

export { switchLanguage, initializeLanguage, t };