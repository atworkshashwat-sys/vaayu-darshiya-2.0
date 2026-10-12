// MobiusHero.tsx — Polished Three.js Möbius strip
// Clean geometry, no extraneous meshes. Just the strip, particles, and orbital balls.

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { token, toRgb } from "@/lib/palette";

// Warm token colours for the 3D scene (read from styles.css at runtime).
const rgb01 = (name: string): [number, number, number] =>
  toRgb(token(name)).map((c) => c / 255) as [number, number, number];
const STRIP_FROM = rgb01("sev-moderate"); // --sev-moderate #E2B33C (soft gold)
const STRIP_TO = rgb01("hcho-1"); // --hcho-1 #F3E9D2 (pale cream)

// --- Parametric position ---
function mobiusPoint(u: number, v: number, R = 2.8, w = 0.75): THREE.Vector3 {
  const a = u * Math.PI * 2;
  const x = (R + v * Math.cos(a / 2)) * Math.cos(a);
  const y = (R + v * Math.cos(a / 2)) * Math.sin(a);
  const z = v * Math.sin(a / 2) * w * 4;
  return new THREE.Vector3(x, y, z);
}

// --- Geometry (module scope) ---
function buildMobiusGeometry(): THREE.BufferGeometry {
  const segU = 220;
  const segV = 28;
  const positions: number[] = [];
  const indices: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const colors: number[] = [];

  for (let i = 0; i <= segU; i++) {
    for (let j = 0; j <= segV; j++) {
      const u = i / segU;
      const v = (j / segV - 0.5) * 0.92;
      const p = mobiusPoint(u, v);
      positions.push(p.x, p.y, p.z);
      uvs.push(u, j / segV);

      // Vertex colour gradient — soft gold → pale cream along the strip, with a
      // cream highlight at the edge (all token colours).
      const t = u;
      const edge = 1 - Math.abs(v) * 2.2;
      const r = STRIP_FROM[0] + (STRIP_TO[0] - STRIP_FROM[0]) * t + edge * 0.06;
      const g = STRIP_FROM[1] + (STRIP_TO[1] - STRIP_FROM[1]) * t + edge * 0.1;
      const b = STRIP_FROM[2] + (STRIP_TO[2] - STRIP_FROM[2]) * t + edge * 0.12;
      colors.push(r, g, b);

      const eps = 0.0004;
      const pu = mobiusPoint(u + eps, v);
      const pv = mobiusPoint(u, v + eps);
      const du = pu.clone().sub(p);
      const dv = pv.clone().sub(p);
      const n = du.cross(dv).normalize();
      normals.push(n.x, n.y, n.z);
    }
  }

  for (let i = 0; i < segU; i++) {
    for (let j = 0; j < segV; j++) {
      const a = i * (segV + 1) + j;
      const b = a + segV + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  return geometry;
}

const mobiusGeometry = buildMobiusGeometry();

// --- Particles ---
const PARTICLE_COUNT = 180;
const particleData = {
  positions: new Float32Array(PARTICLE_COUNT * 3),
  speeds: new Float32Array(PARTICLE_COUNT),
  phases: new Float32Array(PARTICLE_COUNT),
  radii: new Float32Array(PARTICLE_COUNT),
};

for (let i = 0; i < PARTICLE_COUNT; i++) {
  const angle = Math.random() * Math.PI * 2;
  const radius = 2.8 + (Math.random() - 0.5) * 3.6;
  particleData.positions[i * 3] = Math.cos(angle) * radius;
  particleData.positions[i * 3 + 1] = Math.sin(angle) * radius;
  particleData.positions[i * 3 + 2] = (Math.random() - 0.5) * 2;
  particleData.speeds[i] = 0.15 + Math.random() * 0.45;
  particleData.phases[i] = Math.random() * Math.PI * 2;
  particleData.radii[i] = radius;
}

// --- Components ---

function MobiusStrip() {
  const group = useRef<THREE.Group>(null);

  useFrame((_, dt) => {
    if (group.current) {
      group.current.rotation.y += dt * 0.25;
      group.current.rotation.z = Math.sin(group.current.rotation.y * 0.4) * 0.035;
    }
  });

  return (
    <group ref={group} rotation={[Math.PI / 2.4, 0, 0]}>
      {/* Main solid strip */}
      <mesh geometry={mobiusGeometry}>
        <meshPhysicalMaterial
          vertexColors
          metalness={0.45}
          roughness={0.32}
          clearcoat={1}
          clearcoatRoughness={0.07}
          side={THREE.DoubleSide}
          emissive={token("hcho-2")}
          emissiveIntensity={0.12}
          envMapIntensity={1.8}
          sheen={0.6}
          sheenColor={token("hcho-2")}
          sheenRoughness={0.2}
        />
      </mesh>

      {/* Thin wireframe accent */}
      <mesh geometry={mobiusGeometry}>
        <meshBasicMaterial
          color={token("sev-moderate")}
          wireframe
          transparent
          opacity={0.07}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      <GoldenBall />
      <TrailBall />
      <ThirdOrb />
      <Particles />
    </group>
  );
}

function GoldenBall() {
  const ref = useRef<THREE.Mesh>(null);
  const glow = useRef<THREE.PointLight>(null);
  const tRef = useRef(0);

  useFrame((_, dt) => {
    tRef.current += dt * 0.08;
    const u = tRef.current % 2;
    const p = mobiusPoint(u, 0);
    if (ref.current) ref.current.position.copy(p);
    if (glow.current) glow.current.position.copy(p);
  });

  return (
    <>
      <mesh ref={ref}>
        <sphereGeometry args={[0.24, 32, 32]} />
        <meshPhysicalMaterial
          color={token("sev-moderate")}
          emissive={token("hcho-3")}
          emissiveIntensity={1.4}
          metalness={1}
          roughness={0.06}
          clearcoat={1}
          clearcoatRoughness={0.02}
          envMapIntensity={2.5}
        />
      </mesh>
      <pointLight ref={glow} color={token("sev-moderate")} intensity={4} distance={5} decay={2} />
    </>
  );
}

function TrailBall() {
  const ref = useRef<THREE.Mesh>(null);
  const tRef = useRef(0.7);

  useFrame((_, dt) => {
    tRef.current += dt * 0.08;
    const u = tRef.current % 2;
    const p = mobiusPoint(u, 0);
    if (ref.current) ref.current.position.copy(p);
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.14, 24, 24]} />
      <meshPhysicalMaterial
        color={token("hcho-3")}
        emissive={token("sev-moderate")}
        emissiveIntensity={1.2}
        metalness={0.95}
        roughness={0.12}
        clearcoat={1}
      />
    </mesh>
  );
}

function ThirdOrb() {
  const ref = useRef<THREE.Mesh>(null);
  const tRef = useRef(1.4);

  useFrame((_, dt) => {
    tRef.current += dt * 0.08;
    const u = tRef.current % 2;
    const p = mobiusPoint(u, 0);
    if (ref.current) ref.current.position.copy(p);
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.09, 16, 16]} />
      <meshPhysicalMaterial
        color={token("hcho-2")}
        emissive={token("sev-moderate")}
        emissiveIntensity={1.6}
        metalness={0.9}
        roughness={0.18}
        clearcoat={1}
      />
    </mesh>
  );
}

function Particles() {
  const ref = useRef<THREE.Points>(null);
  const tRef = useRef(0);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(particleData.positions, 3));
    return geo;
  }, []);

  useFrame((_, dt) => {
    if (!ref.current) return;
    tRef.current += dt;
    const positions = ref.current.geometry.attributes.position as THREE.BufferAttribute;
    const arr = positions.array as Float32Array;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const angle = tRef.current * particleData.speeds[i] + particleData.phases[i];
      const r = particleData.radii[i];
      arr[i * 3] = Math.cos(angle) * r;
      arr[i * 3 + 1] = Math.sin(angle) * r;
      arr[i * 3 + 2] = Math.sin(tRef.current * 0.5 + particleData.phases[i]) * 1.2;
    }
    positions.needsUpdate = true;
  });

  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial
        size={0.065}
        color={token("hcho-2")}
        transparent
        opacity={0.5}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// --- Scene ---
function Scene() {
  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[6, 7, 5]} intensity={1.4} color={token("paper-2")} />
      <directionalLight position={[-7, 3, -4]} intensity={0.55} color={token("hcho-3")} />
      <directionalLight position={[0, -5, -6]} intensity={0.35} color={token("sev-moderate")} />
      <spotLight
        position={[0, 10, 0]}
        angle={0.35}
        penumbra={0.9}
        intensity={1.3}
        color={token("hcho-2")}
        distance={25}
      />
      <pointLight position={[0, 0, 0]} intensity={0.5} color={token("hcho-2")} distance={8} decay={2} />
      <fog attach="fog" args={[token("paper"), 18, 42]} />
      <MobiusStrip />
    </>
  );
}

export default function MobiusHero() {
  return (
    <Canvas
      camera={{ position: [0, 0, 11], fov: 45 }}
      dpr={[1, 2]}
      gl={{
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 1.15,
      }}
      style={{ background: "transparent" }}
    >
      <Scene />
    </Canvas>
  );
}
