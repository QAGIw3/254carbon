import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface ForwardCurveSurfaceProps {
  width?: number;
  height?: number;
}

export default function ForwardCurveSurface({ width = 800, height = 500 }: ForwardCurveSurfaceProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const [wireframeMode, setWireframeMode] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(3, 2, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 1;
    controls.maxDistance = 50;
    controls.maxPolarAngle = Math.PI;

    // Enhanced lighting
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 0.5);
    pointLight.position.set(-5, 5, -5);
    scene.add(pointLight);

    // Create surface geometry with LOD and performance optimizations
    const createSurfaceGeometry = (resolution: number = 50) => {
      // Adaptive resolution based on device performance
      const maxResolution = 100;
      const adaptiveResolution = Math.min(resolution, maxResolution);

      const geometry = new THREE.PlaneGeometry(10, 10, adaptiveResolution, adaptiveResolution);
      geometry.rotateX(-Math.PI / 2);

      const position = geometry.attributes.position as THREE.BufferAttribute;

      // Use typed array for better performance
      const positions = position.array as Float32Array;
      for (let i = 0; i < position.count; i++) {
        const x = positions[i * 3];
        const z = positions[i * 3 + 2];
        // More complex demo surface with multiple waves
        const y = Math.sin(x * 0.6) * Math.cos(z * 0.6) * 0.8 +
                  Math.sin(x * 0.3) * Math.cos(z * 0.3) * 0.4;
        positions[i * 3 + 1] = y;
      }

      position.needsUpdate = true;
      geometry.computeVertexNormals();
      return geometry;
    };

    const geometry = createSurfaceGeometry();
    const material = new THREE.MeshStandardMaterial({
      color: 0x3b82f6,
      wireframe: wireframeMode,
      side: THREE.DoubleSide,
      roughness: 0.4,
      metalness: 0.1
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    mesh.frustumCulled = true; // Enable frustum culling for performance
    scene.add(mesh);
    meshRef.current = mesh;

    // Grid helper
    const grid = new THREE.GridHelper(10, 20, 0x888888, 0xdddddd);
    grid.position.y = -0.01; // Slightly below surface to avoid z-fighting
    scene.add(grid);

    // Axes helpers for orientation
    const axesHelper = new THREE.AxesHelper(2);
    scene.add(axesHelper);

    let anim = true;
    function animate() {
      if (!anim) return;
      requestAnimationFrame(animate);

      if (autoRotate) {
        mesh.rotation.z += 0.005;
      }

      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !renderer) return;
      const newWidth = containerRef.current.clientWidth;
      const newHeight = containerRef.current.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      anim = false;
      controls.dispose();
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      window.removeEventListener('resize', handleResize);
      if (containerRef.current) containerRef.current.innerHTML = '';
    };
  }, [width, height]);

  // Update wireframe mode
  useEffect(() => {
    if (meshRef.current) {
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      material.wireframe = wireframeMode;
    }
  }, [wireframeMode]);

  const toggleWireframe = () => setWireframeMode(!wireframeMode);
  const toggleAutoRotate = () => setAutoRotate(!autoRotate);

  return (
    <div className="relative">
      <div ref={containerRef} style={{ width, height }} />

      {/* Controls */}
      <div className="flex gap-2 mt-4">
        <button
          className={`px-3 py-1 text-sm border rounded hover:bg-gray-50 ${wireframeMode ? 'bg-blue-50 border-blue-200' : ''}`}
          onClick={toggleWireframe}
        >
          {wireframeMode ? 'Solid' : 'Wireframe'}
        </button>
        <button
          className={`px-3 py-1 text-sm border rounded hover:bg-gray-50 ${autoRotate ? 'bg-green-50 border-green-200' : ''}`}
          onClick={toggleAutoRotate}
        >
          {autoRotate ? 'Stop Rotation' : 'Auto Rotate'}
        </button>
        <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
          Reset View
        </button>
      </div>

      {/* Legend */}
      <div className="mt-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <span>Features:</span>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-200 rounded"></div>
            <span>Surface Height</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-gray-200 rounded"></div>
            <span>Grid</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-200 rounded"></div>
            <span>X-Axis</span>
          </div>
        </div>
      </div>
    </div>
  );
}


