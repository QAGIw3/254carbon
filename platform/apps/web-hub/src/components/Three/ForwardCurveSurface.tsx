import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

interface SurfacePoint { x: number; y: number; z: number }

interface ForwardCurveSurfaceProps {
  points?: SurfacePoint[];
  width?: number;
  height?: number;
}

export default function ForwardCurveSurface({ points = [], width = 800, height = 500 }: ForwardCurveSurfaceProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(3, 2, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 10, 7.5);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    // Build a sample surface if no points provided
    const gridSize = 40;
    const geometry = new THREE.PlaneGeometry(10, 10, gridSize, gridSize);
    geometry.rotateX(-Math.PI / 2);

    const position = geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i);
      const z = position.getZ(i);
      const y = Math.sin(x * 0.6) * Math.cos(z * 0.6) * 0.8; // demo surface
      position.setY(i, y);
    }
    position.needsUpdate = true;
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({ color: 0x3b82f6, wireframe: false, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const grid = new THREE.GridHelper(10, 10, 0x888888, 0xdddddd);
    scene.add(grid);

    let anim = true;
    function animate() {
      if (!anim) return;
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      anim = false;
      controls.dispose();
      renderer.dispose();
      if (containerRef.current) containerRef.current.innerHTML = '';
    };
  }, [width, height]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>3D Forward Curve Surface (Three.js)</CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={containerRef} style={{ width, height }} />
      </CardContent>
    </Card>
  );
}


