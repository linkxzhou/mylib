import * as THREE from 'three';

class Scene3DManager {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.objects = [];
        this.animationId = null;
        this.isAnimating = false;
        this.frameCount = 0;
        this.lastTime = performance.now();
        
        this.init();
    }

    init() {
        // 创建场景
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x222222);

        // 创建相机
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);

        // 创建渲染器
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // 添加光源
        this.setupLights();

        // 添加网格地面
        this.addGrid();

        // 添加一些初始对象
        this.addInitialObjects();

        // 设置控制器
        this.setupControls();

        // 开始渲染循环
        this.startAnimation();

        // 监听窗口大小变化
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLights() {
        // 环境光
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        // 方向光
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        // 点光源
        const pointLight = new THREE.PointLight(0xff6600, 0.5, 50);
        pointLight.position.set(-10, 10, -10);
        this.scene.add(pointLight);
    }

    addGrid() {
        const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0x444444);
        this.scene.add(gridHelper);

        // 添加坐标轴
        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);
    }

    addInitialObjects() {
        // 添加一个旋转的立方体
        this.addCube(0, 1, 0, 0xff6600);
        
        // 添加一个球体
        this.addSphere(-3, 1, 0, 0x00ff66);
        
        // 添加一个圆环
        this.addTorus(3, 1, 0, 0x6600ff);
    }

    setupControls() {
        // 简单的鼠标控制
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let targetRotationX = 0;
        let targetRotationY = 0;
        let currentRotationX = 0;
        let currentRotationY = 0;

        this.renderer.domElement.addEventListener('mousedown', (event) => {
            isMouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });

        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (isMouseDown) {
                const deltaX = event.clientX - mouseX;
                const deltaY = event.clientY - mouseY;
                
                targetRotationY += deltaX * 0.01;
                targetRotationX += deltaY * 0.01;
                
                mouseX = event.clientX;
                mouseY = event.clientY;
            }
        });

        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
        });

        // 鼠标滚轮缩放
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const scale = event.deltaY > 0 ? 1.1 : 0.9;
            this.camera.position.multiplyScalar(scale);
            event.preventDefault();
        });

        // 在动画循环中更新相机旋转
        this.updateCameraRotation = () => {
            currentRotationX += (targetRotationX - currentRotationX) * 0.05;
            currentRotationY += (targetRotationY - currentRotationY) * 0.05;
            
            const radius = this.camera.position.length();
            this.camera.position.x = radius * Math.sin(currentRotationY) * Math.cos(currentRotationX);
            this.camera.position.y = radius * Math.sin(currentRotationX);
            this.camera.position.z = radius * Math.cos(currentRotationY) * Math.cos(currentRotationX);
            this.camera.lookAt(0, 0, 0);
        };
    }

    addCube(x = 0, y = 0, z = 0, color = 0xff6600) {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshPhongMaterial({ color });
        const cube = new THREE.Mesh(geometry, material);
        
        cube.position.set(x, y, z);
        cube.castShadow = true;
        cube.receiveShadow = true;
        
        // 添加旋转动画属性
        cube.userData = {
            type: 'cube',
            rotationSpeed: { x: 0.01, y: 0.02, z: 0.005 }
        };
        
        this.scene.add(cube);
        this.objects.push(cube);
        
        return cube;
    }

    addSphere(x = 0, y = 0, z = 0, color = 0x00ff66) {
        const geometry = new THREE.SphereGeometry(0.8, 32, 32);
        const material = new THREE.MeshPhongMaterial({ color });
        const sphere = new THREE.Mesh(geometry, material);
        
        sphere.position.set(x, y, z);
        sphere.castShadow = true;
        sphere.receiveShadow = true;
        
        // 添加浮动动画属性
        sphere.userData = {
            type: 'sphere',
            floatSpeed: 0.02,
            floatHeight: 0.5,
            originalY: y
        };
        
        this.scene.add(sphere);
        this.objects.push(sphere);
        
        return sphere;
    }

    addTorus(x = 0, y = 0, z = 0, color = 0x6600ff) {
        const geometry = new THREE.TorusGeometry(0.8, 0.3, 16, 100);
        const material = new THREE.MeshPhongMaterial({ color });
        const torus = new THREE.Mesh(geometry, material);
        
        torus.position.set(x, y, z);
        torus.castShadow = true;
        torus.receiveShadow = true;
        
        // 添加旋转动画属性
        torus.userData = {
            type: 'torus',
            rotationSpeed: { x: 0.02, y: 0.01, z: 0.03 }
        };
        
        this.scene.add(torus);
        this.objects.push(torus);
        
        return torus;
    }

    clearScene() {
        // 移除所有用户添加的对象
        this.objects.forEach(obj => {
            this.scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        });
        this.objects = [];
    }

    resetCamera() {
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);
    }

    startAnimation() {
        if (!this.isAnimating) {
            this.isAnimating = true;
            this.animate();
        }
    }

    stopAnimation() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    toggleAnimation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
        return this.isAnimating;
    }

    animate() {
        if (!this.isAnimating) return;
        
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // 更新对象动画
        this.objects.forEach(obj => {
            if (obj.userData.type === 'cube' || obj.userData.type === 'torus') {
                obj.rotation.x += obj.userData.rotationSpeed.x;
                obj.rotation.y += obj.userData.rotationSpeed.y;
                obj.rotation.z += obj.userData.rotationSpeed.z;
            } else if (obj.userData.type === 'sphere') {
                const time = Date.now() * obj.userData.floatSpeed;
                obj.position.y = obj.userData.originalY + Math.sin(time) * obj.userData.floatHeight;
            }
        });
        
        // 更新相机控制
        if (this.updateCameraRotation) {
            this.updateCameraRotation();
        }
        
        // 计算FPS
        this.frameCount++;
        const currentTime = performance.now();
        if (currentTime - this.lastTime >= 1000) {
            const fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
            this.updateFPS(fps);
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
        
        // 更新对象计数
        this.updateObjectCount();
        
        // 渲染场景
        this.renderer.render(this.scene, this.camera);
    }

    updateFPS(fps) {
        const fpsElement = document.getElementById('fps');
        if (fpsElement) {
            fpsElement.textContent = fps;
        }
    }

    updateObjectCount() {
        const countElement = document.getElementById('object-count');
        if (countElement) {
            countElement.textContent = this.objects.length;
        }
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    dispose() {
        this.stopAnimation();
        
        // 清理资源
        this.clearScene();
        
        if (this.renderer) {
            this.renderer.dispose();
            if (this.container && this.renderer.domElement) {
                this.container.removeChild(this.renderer.domElement);
            }
        }
        
        window.removeEventListener('resize', this.onWindowResize);
    }
}

export { Scene3DManager };
export default Scene3DManager;