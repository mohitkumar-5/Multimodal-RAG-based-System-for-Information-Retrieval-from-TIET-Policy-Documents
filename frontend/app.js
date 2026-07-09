// ==========================================================================
// CONFIGURATION & GLOBAL STATE
// ==========================================================================
const BACKEND_URL = ""; 

let sessionId = "";
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let attachedImageFile = null;

// ==========================================================================
// INITIALIZATION
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    initSession();
    setupEventListeners();
    fetchFeedbackSummary();
    checkApiStatus();
    initFluidBackground();
    initLiveTerminalStream();
    initScrollAnimations();
    
    // Onboarding welcome message
    setTimeout(appendWelcomeMessage, 600);
    
    // Auto-poll feedback summary every 30 seconds
    setInterval(fetchFeedbackSummary, 30000);
});

function appendWelcomeMessage() {
    const container = document.getElementById("chat-messages-container");
    if (!container || container.children.length > 0) return;
    appendBotBubble("Hey! 👋 I'm PolicyLens, your TIET academic assistant. How can I help you today?");
}

// Helper to determine the API root dynamically
function getApiUrl(endpoint) {
    if (BACKEND_URL) {
        return `${BACKEND_URL.replace(/\/$/, '')}${endpoint}`;
    }
    return endpoint;
}

// Generate or retrieve persistent Session ID
function initSession() {
    let savedSession = localStorage.getItem("tiet_session_id");
    if (!savedSession) {
        savedSession = "tiet_session_" + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        localStorage.setItem("tiet_session_id", savedSession);
    }
    sessionId = savedSession;
    const sessionDisplay = document.getElementById("session-id-display");
    if (sessionDisplay) {
        sessionDisplay.innerText = sessionId.substring(0, 15) + "...";
    }
}

// ==========================================================================
// PAGE / TAB NAVIGATION
// ==========================================================================
window.switchTab = function(tabId) {
    document.querySelectorAll('.page-view').forEach(view => view.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    
    const targetView = document.getElementById(`view-${tabId}`);
    const targetLink = document.getElementById(`tab-${tabId}`);
    if (targetView) targetView.classList.add('active');
    if (targetLink) targetLink.classList.add('active');
    
    // Re-check elements visibility when switching pages
    setTimeout(initScrollAnimations, 100);
};

window.setQueryAndLaunch = function(text) {
    switchTab('chat');
    const input = document.getElementById("chat-input");
    if (input) {
        input.value = text;
        input.style.height = "auto";
        input.style.height = (input.scrollHeight - 4) + "px";
        input.focus();
    }
};

window.scrollToFeatures = function() {
    const section = document.getElementById("features-section");
    if (section) {
        section.scrollIntoView({ behavior: "smooth" });
    }
};

// ==========================================================================
// SCROLL ENTRANCE INTERSECTION OBSERVATION
// ==========================================================================
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("in-view");
            }
        });
    }, {
        threshold: 0.08,
        rootMargin: "0px 0px -40px 0px"
    });
    
    document.querySelectorAll(".animate-on-scroll").forEach(el => {
        observer.observe(el);
    });
}

// ==========================================================================
// INTERACTIVE WEBGL JELLYFISH BACKGROUND (THREE.JS + SHADERS + PARTICLES)
// ==========================================================================
let renderer, scene, camera, blobMesh, material;
let particleSystem, particlePositions, particleData = [];
const particleCount = 80;
let mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
let clock = new THREE.Clock();

// GLSL Vertex Shader: Displacement using 3D Simplex Noise for organic morphing
const vertexShader = `
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
    
    float snoise(vec3 v) {
        const vec2 C = vec2(1.0/6.0, 1.0/3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
        
        vec3 i  = floor(v + dot(v, C.yyy) );
        vec3 x0 = v - i + dot(i, C.xxx) ;
        
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min( g.xyz, l.zxy );
        vec3 i2 = max( g.xyz, l.zxy );
        
        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;
        
        i = mod289(i);
        vec4 p = permute( permute( permute(
                    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
                + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
                
        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;
        
        vec4 j = p - 49.0 * floor(p * ns.z);
        
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_ );
        
        vec4 x = x_ *ns.x + ns.yyyy;
        vec4 y = y_ *ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        
        vec4 b0 = vec4( x.xy, y.xy );
        vec4 b1 = vec4( x.zw, y.zw );
        
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
        
        vec3 p0 = vec3(a0.xy,h.x);
        vec3 p1 = vec3(a0.zw,h.y);
        vec3 p2 = vec3(a1.xy,h.z);
        vec3 p3 = vec3(a1.zw,h.w);
        
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                        dot(p2,x2), dot(p3,x3) ) );
    }

    varying vec3 vNormal;
    varying vec3 vPosition;
    varying float vNoise;
    uniform float uTime;

    void main() {
        vNormal = normal;
        vPosition = position;
        
        // Displace position based on simplex noise (morphing blob)
        float noise = snoise(position * 1.4 + vec3(0.0, 0.0, uTime * 0.75));
        vNoise = noise;
        
        vec3 displaced = position + normal * noise * 0.35;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
    }
`;

// GLSL Fragment Shader: Glowing Fresnel rim lighting blending neon purple, green, and cyan colors
const fragmentShader = `
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying float vNoise;
    uniform float uTime;

    void main() {
        vec3 normal = normalize(vNormal);
        vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));
        
        float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
        
        vec3 neonPurple = vec3(0.65, 0.15, 0.95);
        vec3 neonGreen = vec3(0.05, 0.95, 0.35);
        vec3 neonCyan = vec3(0.0, 0.85, 1.0);
        
        vec3 color = mix(neonPurple, neonGreen, vNoise * 0.5 + 0.5);
        color = mix(color, neonCyan, fresnel);
        
        vec3 finalColor = color + (neonCyan * fresnel * 0.6) + (neonGreen * max(0.0, vNoise) * 0.15);
        float alpha = clamp(0.18 + fresnel * 0.72 + max(0.0, vNoise) * 0.25, 0.0, 0.85);
        
        gl_FragColor = vec4(finalColor, alpha);
    }
`;

function initFluidBackground() {
    const canvas = document.getElementById("fluid-canvas");
    if (!canvas) return;

    scene = new THREE.Scene();
    
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.z = 8;
    
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        alpha: true,
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // 1. Mesh setup: Sphere geometry + custom shader material (detail reduced to 5 for high performance)
    const geometry = new THREE.IcosahedronGeometry(1.6, 5);
    
    material = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms: {
            uTime: { value: 0.0 }
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending
    });
    
    blobMesh = new THREE.Mesh(geometry, material);
    scene.add(blobMesh);

    // 2. Setup glowing jellyfish trail particles (tentacle sparks flowing downwards)
    const pGeometry = new THREE.BufferGeometry();
    particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
        particlePositions[i * 3] = (Math.random() - 0.5) * 8;
        particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 8;
        particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 2;
        
        particleData.push({
            x: (Math.random() - 0.5) * 0.03,
            y: -(Math.random() * 0.08 + 0.04), // sink downwards
            z: (Math.random() - 0.5) * 0.03,
            life: Math.random(),
            decay: Math.random() * 0.015 + 0.008
        });
    }
    pGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
    
    const pMaterial = new THREE.PointsMaterial({
        color: 0x06b6d4, // Cyan neon
        size: 0.07,
        transparent: true,
        opacity: 0.45,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
    
    particleSystem = new THREE.Points(pGeometry, pMaterial);
    scene.add(particleSystem);

    // 3. Pointer move interaction
    window.addEventListener("pointermove", (e) => {
        mouse.targetX = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.targetY = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    window.addEventListener("resize", onWindowResize);
    requestAnimationFrame(renderFluidAnimation);
}

function onWindowResize() {
    if (!camera || !renderer) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function renderFluidAnimation() {
    const delta = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();

    if (material) {
        material.uniforms.uTime.value = elapsedTime;
    }

    if (blobMesh) {
        const aspect = window.innerWidth / window.innerHeight;
        const viewportHeight = 2.0 * Math.tan((camera.fov * Math.PI) / 360.0) * camera.position.z;
        const viewportWidth = viewportHeight * aspect;

        const targetX = mouse.targetX * (viewportWidth / 2.0);
        const targetY = mouse.targetY * (viewportHeight / 2.0);

        // Smooth follow movement (lerping coordinates)
        blobMesh.position.x += (targetX - blobMesh.position.x) * 0.06;
        blobMesh.position.y += (targetY - blobMesh.position.y) * 0.06;
        
        blobMesh.rotation.y += delta * 0.15;
        blobMesh.rotation.z += delta * 0.08;
    }

    // Update trailing jellyfish stinger particles
    if (particleSystem && blobMesh) {
        const posArr = particlePositions;
        for (let i = 0; i < particleCount; i++) {
            const data = particleData[i];
            data.life -= data.decay;
            
            posArr[i * 3] += data.x;
            posArr[i * 3 + 1] += data.y;
            posArr[i * 3 + 2] += data.z;
            
            // Respawn particle at blob mesh position when dead
            if (data.life <= 0) {
                data.life = 1.0;
                posArr[i * 3] = blobMesh.position.x + (Math.random() - 0.5) * 1.2;
                posArr[i * 3 + 1] = blobMesh.position.y - 0.4 + (Math.random() - 0.5) * 0.4; // spawn under jellyfish body
                posArr[i * 3 + 2] = blobMesh.position.z + (Math.random() - 0.5) * 0.6;
                
                data.x = (Math.random() - 0.5) * 0.03;
                data.y = -(Math.random() * 0.06 + 0.03); // float down
                data.z = (Math.random() - 0.5) * 0.03;
            }
        }
        particleSystem.geometry.attributes.position.needsUpdate = true;
    }

    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }

    requestAnimationFrame(renderFluidAnimation);
}

// ==========================================================================
// LIVE PLATFORM REQUEST LOGS TERMINAL (TILANTRA STYLE)
// ==========================================================================
const mockRequestLogs = [
    { type: 'Routed', query: 'What is the CS admission criteria?', outcome: 'served via Qwen-7B (420ms)', style: 'outcome-purple', pill: 'pill-routed' },
    { type: 'Cached', query: 'Show UG ECE course schemes', outcome: 'matched Redis session (12ms)', style: 'outcome-purple', pill: 'pill-caching' },
    { type: 'MMR', query: 'Find hostel room guidelines', outcome: 'extracted 5 source PDFs (190ms)', style: 'outcome-blue', pill: 'pill-retrieval' },
    { type: 'TTS', query: 'Grade criteria for PhD candidate', outcome: 'generated MP3 audio (8.2kb)', style: 'outcome-success', pill: 'pill-tts' },
    { type: 'OCR', query: 'Analyze policy structure image', outcome: 'processed Vision layout (310ms)', style: 'outcome-blue', pill: 'pill-ocr' },
    { type: 'Routed', query: 'What is the tuition fee for MCA?', outcome: 'served via Groq-8B fallback (280ms)', style: 'outcome-success', pill: 'pill-routed' },
    { type: 'Cached', query: 'How to apply for lateral entry?', outcome: 'loaded memory thread (8ms)', style: 'outcome-purple', pill: 'pill-caching' },
    { type: 'MMR', query: 'Hostel fee chart and dues 2026', outcome: 'searched Qdrant index (145ms)', style: 'outcome-blue', pill: 'pill-retrieval' }
];

function initLiveTerminalStream() {
    const tbody = document.getElementById("live-requests-tbody");
    const speedIndicator = document.getElementById("req-speed-indicator");
    if (!tbody) return;

    // Periodically push simulated logs
    setInterval(() => {
        const log = mockRequestLogs[Math.floor(Math.random() * mockRequestLogs.length)];
        
        const tr = document.createElement("tr");
        tr.className = "log-row";
        tr.innerHTML = `
            <td><span class="type-pill ${log.pill}">${log.type}</span></td>
            <td class="log-query">${escapeHtml(log.query)}</td>
            <td class="log-outcome ${log.style}">${log.outcome}</td>
        `;
        
        tbody.appendChild(tr);

        while (tbody.rows.length > 4) {
            tbody.rows[0].remove();
        }
        
        if (speedIndicator) {
            const count = Math.floor(Math.random() * 200) + 850;
            speedIndicator.innerText = `${count} req/min`;
        }
    }, 2500);
}

// ==========================================================================
// EVENT LISTENERS REGISTER
// ==========================================================================
function setupEventListeners() {
    const resetBtn = document.getElementById("reset-session-btn");
    if (resetBtn) {
        resetBtn.addEventListener("click", resetSession);
    }

    const chatInput = document.getElementById("chat-input");
    if (chatInput) {
        chatInput.addEventListener("input", function() {
            this.style.height = "auto";
            this.style.height = (this.scrollHeight - 4) + "px";
        });

        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitMessage();
            }
        });
    }

    const sendBtn = document.getElementById("send-message-btn");
    if (sendBtn) {
        sendBtn.addEventListener("click", submitMessage);
    }

    const stopBtn = document.getElementById("stop-generation-btn");
    if (stopBtn) {
        stopBtn.addEventListener("click", window.stopGeneration);
    }

    const fileInput = document.getElementById("image-file-input");
    const uploadBtn = document.getElementById("image-upload-btn");
    const removeImgBtn = document.getElementById("remove-image-preview-btn");
    
    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener("click", () => fileInput.click());
    }
    if (fileInput) {
        fileInput.addEventListener("change", handleImageSelect);
    }
    if (removeImgBtn) {
        removeImgBtn.addEventListener("click", removeAttachedImage);
    }

    const voiceBtn = document.getElementById("voice-record-btn");
    if (voiceBtn) {
        voiceBtn.addEventListener("click", toggleVoiceRecording);
    }
}

// ==========================================================================
// SUGGESTION CARDS HELPER
// ==========================================================================
window.setQuery = function(text) {
    const input = document.getElementById("chat-input");
    if (input) {
        input.value = text;
        input.style.height = "auto";
        input.style.height = (input.scrollHeight - 4) + "px";
        input.focus();
    }
};

// ==========================================================================
// CLIENT RUNTIME / API CONNECTION VERIFIER
// ==========================================================================
async function checkApiStatus() {
    const text = document.getElementById("conn-indicator-text");
    try {
        const response = await fetch(getApiUrl("/api/health"));
        if (response.ok && text) {
            text.innerText = "API Online";
        } else {
            throw new Error();
        }
    } catch {
        if (text) {
            text.innerText = "API Offline";
        }
    }
}

// ==========================================================================
// RESET CONVERSATION STATE
// ==========================================================================
function resetSession() {
    if (confirm("Are you sure you want to start a new session? This will clear current chat history.")) {
        localStorage.removeItem("tiet_session_id");
        initSession();
        
        const chatContainer = document.getElementById("chat-messages-container");
        if (chatContainer) {
            chatContainer.innerHTML = "";
        }
        
        removeAttachedImage();
        const chatInput = document.getElementById("chat-input");
        if (chatInput) {
            chatInput.value = "";
        }
        fetchFeedbackSummary();
        setTimeout(appendWelcomeMessage, 300);
    }
}

// ==========================================================================
// ATTACHED IMAGE HANDLER
// ==========================================================================
function handleImageSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image file.");
        return;
    }

    attachedImageFile = file;
    
    const reader = new FileReader();
    reader.onload = function(event) {
        document.getElementById("image-preview-thumbnail").src = event.target.result;
        document.getElementById("image-preview-tray").style.display = "flex";
        document.getElementById("image-upload-btn").classList.add("active");
    };
    reader.readAsDataURL(file);
}

function removeAttachedImage() {
    attachedImageFile = null;
    const fileInput = document.getElementById("image-file-input");
    if (fileInput) fileInput.value = "";
    
    document.getElementById("image-preview-tray").style.display = "none";
    document.getElementById("image-preview-thumbnail").src = "";
    document.getElementById("image-upload-btn").classList.remove("active");
}

// ==========================================================================
// AUDIO VOICE CAPTURE HANDLER
// ==========================================================================
async function toggleVoiceRecording() {
    const recordBtn = document.getElementById("voice-record-btn");
    const pulseWave = document.getElementById("mic-pulse-wave");
    const micIcon = document.getElementById("mic-icon-svg");

    if (isRecording) {
        isRecording = false;
        if (recordBtn) recordBtn.classList.remove("recording");
        if (pulseWave) pulseWave.style.display = "none";
        if (micIcon) micIcon.style.display = "block";
        
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = uploadVoiceRecording;
            
            isRecording = true;
            if (recordBtn) recordBtn.classList.add("recording");
            if (pulseWave) pulseWave.style.display = "block";
            if (micIcon) micIcon.style.display = "none";
            
            mediaRecorder.start();
        } catch (err) {
            console.error("Audio recording permission denied or failed:", err);
            alert("Unable to access microphone. Please check system permissions.");
        }
    }
}

// ==========================================================================
// SUBMIT CONVERSATION REQUESTS
// ==========================================================================
async function submitMessage() {
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    
    if (!text && !attachedImageFile) return;

    if (attachedImageFile) {
        const thumbnailSrc = document.getElementById("image-preview-thumbnail").src;
        appendUserBubble(text, thumbnailSrc);
        
        const imageToUpload = attachedImageFile;
        const textQuery = text;
        removeAttachedImage();
        input.value = "";
        input.style.height = "auto";
        
        await sendImageQuery(imageToUpload, textQuery);
    } else {
        appendUserBubble(text);
        input.value = "";
        input.style.height = "auto";
        
        await sendTextQuery(text);
    }
}

async function sendTextQuery(question) {
    const botRowId = appendBotLoader();
    
    try {
        const response = await fetch(getApiUrl("/api/chat/text"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, session_id: sessionId })
        });

        const data = await response.json();
        removeBotLoader(botRowId);

        if (response.ok && data.success) {
            appendBotBubble(data.answer, data.feedback_id);
        } else {
            const errMsg = data.detail || data.error || "An error occurred on the server.";
            appendBotBubble(`⚠️ Error: ${errMsg}`);
        }
    } catch (err) {
        console.error(err);
        removeBotLoader(botRowId);
        appendBotBubble("⚠️ Failed to reach the server. Please check your internet connection.");
    }
}

async function sendImageQuery(file, textPrompt) {
    const botRowId = appendBotLoader();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId);
    if (textPrompt) {
        formData.append("prompt", textPrompt);
    }

    try {
        const response = await fetch(getApiUrl("/api/chat/image"), {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        removeBotLoader(botRowId);

        if (response.ok && data.success) {
            const formattedText = `**Understood Topic:** *${data.extracted_query}*\n\n${data.answer}`;
            appendBotBubble(formattedText, data.feedback_id);
        } else {
            const errMsg = data.detail || data.error || "Failed to process image.";
            appendBotBubble(`⚠️ Error: ${errMsg}`);
        }
    } catch (err) {
        console.error(err);
        removeBotLoader(botRowId);
        appendBotBubble("⚠️ Failed to submit image. Please try again.");
    }
}

async function uploadVoiceRecording() {
    if (audioChunks.length === 0) return;

    const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
    const fileExtension = audioBlob.type.split("/")[1]?.split(";")[0] || "webm";
    const audioFile = new File([audioBlob], `recording.${fileExtension}`, { type: audioBlob.type });

    const userRowId = appendUserBubble("🎙️ *Sending voice recording...*");
    const botRowId = appendBotLoader();

    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("session_id", sessionId);

    try {
        const response = await fetch(getApiUrl("/api/chat/voice"), {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        removeBotLoader(botRowId);

        if (response.ok && data.success) {
            document.getElementById(userRowId).querySelector(".message-bubble").innerHTML = `🎙️ "${data.transcribed_question}"`;
            appendBotBubble(data.answer, data.feedback_id, data.audio_base64);
        } else {
            removeBotLoader(botRowId);
            document.getElementById(userRowId).querySelector(".message-bubble").innerHTML = `🎙️ *(Voice submission failed)*`;
            const errMsg = data.detail || data.error || "Audio transcription failed.";
            appendBotBubble(`⚠️ Error: ${errMsg}`);
        }
    } catch (err) {
        console.error(err);
        removeBotLoader(botRowId);
        document.getElementById(userRowId).querySelector(".message-bubble").innerHTML = `🎙️ *(Voice submission failed)*`;
        appendBotBubble("⚠️ Failed to upload audio recording. Please try again.");
    }
}

// ==========================================================================
// FEEDBACK API CALLS
// ==========================================================================
async function submitFeedback(feedbackId, rating, buttonElement) {
    try {
        const response = await fetch(getApiUrl("/api/feedback"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ feedback_id: feedbackId, rating })
        });
        
        if (response.ok) {
            const parent = buttonElement.parentElement;
            parent.classList.add("rated");
            
            const upBtn = parent.querySelector(".feedback-btn.up");
            const downBtn = parent.querySelector(".feedback-btn.down");
            
            upBtn.classList.remove("active");
            downBtn.classList.remove("active");
            
            if (rating === "up") {
                upBtn.classList.add("active");
            } else {
                downBtn.classList.add("active");
            }
            fetchFeedbackSummary();
        }
    } catch (err) {
        console.error("Failed to submit feedback:", err);
    }
}

async function fetchFeedbackSummary() {
    try {
        const response = await fetch(getApiUrl("/api/feedback/summary"));
        const data = await response.json();
        
        if (response.ok) {
            const upElem = document.getElementById("feedback-summary-up");
            const downElem = document.getElementById("feedback-summary-down");
            if (upElem) upElem.innerText = data.thumbs_up;
            if (downElem) downElem.innerText = data.thumbs_down;
        }
    } catch (err) {
        console.warn("Feedback stats currently unavailable:", err);
    }
}

// ==========================================================================
// HTML RENDERING HELPERS
// ==========================================================================
function appendUserBubble(text, imageSrc = null) {
    const container = document.getElementById("chat-messages-container");
    const rowId = "msg_" + Date.now() + "_" + Math.floor(Math.random()*1000);
    
    let mediaHtml = "";
    if (imageSrc) {
        mediaHtml = `<div class="message-media-attachment" style="max-width:200px; margin-bottom:8px; border-radius:8px; overflow:hidden;"><img src="${imageSrc}" alt="Attachment" style="width:100%;"></div>`;
    }

    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.id = rowId;
    row.innerHTML = `
        <div class="message-bubble">
            ${mediaHtml}
            <p>${escapeHtml(text)}</p>
        </div>
        <div class="message-avatar">U</div>
    `;
    
    container.appendChild(row);
    scrollToBottom();
    return rowId;
}

const activeAudioPlayers = {};

window.toggleAudioPlayback = function(playerId, base64Data) {
    let player = activeAudioPlayers[playerId];
    const container = document.getElementById(`audio_control_${playerId}`);
    if (!container) return;
    
    const playBtn = container.querySelector(".player-btn.play");
    const stopBtn = container.querySelector(".player-btn.stop");
    const playSvg = playBtn.querySelector(".play-svg");
    const pauseSvg = playBtn.querySelector(".pause-svg");

    if (!player) {
        const audioSrc = `data:audio/mp3;base64,${base64Data}`;
        player = new Audio(audioSrc);
        activeAudioPlayers[playerId] = player;
        player.onended = () => {
            resetPlayerUI(playerId);
        };
    }

    if (player.paused) {
        pauseAllOtherPlayers(playerId);
        player.play();
        playSvg.style.display = "none";
        pauseSvg.style.display = "block";
        stopBtn.removeAttribute("disabled");
    } else {
        player.pause();
        playSvg.style.display = "block";
        pauseSvg.style.display = "none";
    }
};

window.stopAudioPlayback = function(playerId) {
    const player = activeAudioPlayers[playerId];
    if (player) {
        player.pause();
        player.currentTime = 0;
        resetPlayerUI(playerId);
    }
};

window.setAudioSpeed = function(playerId, speed) {
    const player = activeAudioPlayers[playerId];
    if (player) {
        player.playbackRate = speed;
    }
    const container = document.getElementById(`audio_control_${playerId}`);
    if (container) {
        const speedBtns = container.querySelectorAll(".speed-controls .speed-btn");
        speedBtns.forEach(btn => btn.classList.remove("active"));
        
        if (speed === 1.0) {
            document.getElementById(`speed_1_${playerId}`).classList.add("active");
        } else if (speed === 1.25) {
            document.getElementById(`speed_125_${playerId}`).classList.add("active");
        } else if (speed === 1.5) {
            document.getElementById(`speed_15_${playerId}`).classList.add("active");
        }
    }
};

function resetPlayerUI(playerId) {
    const container = document.getElementById(`audio_control_${playerId}`);
    if (container) {
        const playBtn = container.querySelector(".player-btn.play");
        const stopBtn = container.querySelector(".player-btn.stop");
        playBtn.querySelector(".play-svg").style.display = "block";
        playBtn.querySelector(".pause-svg").style.display = "none";
        stopBtn.setAttribute("disabled", "true");
    }
}

function pauseAllOtherPlayers(exceptPlayerId) {
    Object.keys(activeAudioPlayers).forEach(pid => {
        if (pid !== exceptPlayerId) {
            const p = activeAudioPlayers[pid];
            if (p && !p.paused) {
                p.pause();
                resetPlayerUI(pid);
            }
        }
    });
}

function appendBotBubble(rawText, feedbackId = null, audioBase64 = null) {
    const container = document.getElementById("chat-messages-container");
    const rowId = "bot_" + Date.now() + "_" + Math.floor(Math.random()*1000);
    const formattedHtml = parseMarkdown(rawText);
    
    let audioCardHtml = "";
    const playerId = "audio_" + Date.now() + "_" + Math.floor(Math.random()*1000);
    if (audioBase64) {
        audioCardHtml = `
            <div class="audio-player-control" id="audio_control_${playerId}" style="display:flex; align-items:center; gap:8px; margin-top:12px; background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:12px; border:1px solid rgba(255,255,255,0.05);">
                <button class="player-btn play" onclick="toggleAudioPlayback('${playerId}', '${audioBase64}')" title="Play/Pause" style="background:none; border:none; color:#ffffff; cursor:pointer;">
                    <svg class="play-svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                    <svg class="pause-svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor" style="display:none;"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                </button>
                <button class="player-btn stop" onclick="stopAudioPlayback('${playerId}')" title="Stop" disabled style="background:none; border:none; color:#ffffff; cursor:pointer; opacity:0.5;">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>
                </button>
                <div class="speed-controls" style="display:flex; gap:4px; margin-left:auto;">
                    <button class="speed-btn active" id="speed_1_${playerId}" onclick="setAudioSpeed('${playerId}', 1.0)" style="background:none; border:none; color:var(--text-secondary); font-size:11px; cursor:pointer;">1.0x</button>
                    <button class="speed-btn" id="speed_125_${playerId}" onclick="setAudioSpeed('${playerId}', 1.25)" style="background:none; border:none; color:var(--text-secondary); font-size:11px; cursor:pointer;">1.25x</button>
                    <button class="speed-btn" id="speed_15_${playerId}" onclick="setAudioSpeed('${playerId}', 1.5)" style="background:none; border:none; color:var(--text-secondary); font-size:11px; cursor:pointer;">1.5x</button>
                </div>
            </div>
        `;
    }

    let feedbackHtml = "";
    if (feedbackId) {
        feedbackHtml = `
            <div class="message-feedback-row">
                <button class="feedback-btn up" onclick="submitFeedback('${feedbackId}', 'up', this)" title="Helpful">👍</button>
                <button class="feedback-btn down" onclick="submitFeedback('${feedbackId}', 'down', this)" title="Unhelpful">👎</button>
            </div>
        `;
    }

    const row = document.createElement("div");
    row.className = "message-row bot-row";
    row.id = rowId;
    row.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content-wrapper" style="max-width:80%;">
            <div class="message-bubble" id="bubble_content_${rowId}"></div>
            ${feedbackHtml}
        </div>
    `;
    
    container.appendChild(row);
    scrollToBottom();
    
    const contentDiv = document.getElementById(`bubble_content_${rowId}`);
    if (contentDiv) {
        showStopButton();
        typeHtml(contentDiv, formattedHtml, 6, () => {
            hideStopButton();
            if (audioBase64) {
                const tempDiv = document.createElement("div");
                tempDiv.innerHTML = audioCardHtml;
                contentDiv.appendChild(tempDiv.firstElementChild);
                scrollToBottom();
                setTimeout(() => {
                    window.toggleAudioPlayback(playerId, audioBase64);
                }, 50);
            }
        });
    }
}

function appendBotLoader() {
    const container = document.getElementById("chat-messages-container");
    const rowId = "loader_" + Date.now();
    
    const row = document.createElement("div");
    row.className = "message-row bot-row loading-row";
    row.id = rowId;
    row.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content-wrapper">
            <div class="message-bubble" style="padding:12px 18px;">
                <div class="typing-loader">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;
    container.appendChild(row);
    scrollToBottom();
    return rowId;
}

function removeBotLoader(rowId) {
    const loader = document.getElementById(rowId);
    if (loader) {
        loader.remove();
    }
}

function scrollToBottom() {
    const viewport = document.getElementById("chat-viewport");
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
}

// ==========================================================================
// STRING & MARKDOWN PARSERS
// ==========================================================================
function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function parseMarkdown(text) {
    if (!text) return "";
    
    let rawText = text;
    const uniqueSources = new Set();
    
    const sourceBlockRegex = /\(Source:\s*([^\)]+)\)/gi;
    let blockMatch;
    
    while ((blockMatch = sourceBlockRegex.exec(rawText)) !== null) {
        const content = blockMatch[1];
        const fileRegex = /([\w\-\.]+\.pdf)(?:\s*,?\s*(?:Page|p\.)?\s*(\d+))?/gi;
        let fileMatch;
        while ((fileMatch = fileRegex.exec(content)) !== null) {
            const filename = fileMatch[1].trim();
            const page = fileMatch[2] ? fileMatch[2].trim() : "?";
            uniqueSources.add(`${filename} (Page ${page})`);
        }
    }
    
    let cleanText = rawText.replace(sourceBlockRegex, "");
    cleanText = cleanText.replace(/\s+([.,;:?])/g, "$1");
    cleanText = cleanText.replace(/\s{2,}/g, " ");

    let html = "";
    try {
        if (window.marked && typeof window.marked.parse === "function") {
            html = window.marked.parse(cleanText);
        } else {
            html = cleanText;
            html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
            html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
            html = parseTablesInText(html);
            html = html.replace(/\n/g, "<br>");
        }
    } catch (e) {
        console.error("Marked parsing error:", e);
        html = cleanText;
    }
    
    if (uniqueSources.size > 0) {
        html += '<div style="margin-top:14px; border-top:1px solid rgba(255,255,255,0.05); padding-top:10px;">';
        html += '<span style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px;">Sources Consulted:</span><ul style="list-style:none; margin:6px 0 0 0; padding:0; display:flex; flex-direction:column; gap:4px;">';
        uniqueSources.forEach(source => {
            html += `<li style="font-size:12px; color:var(--text-secondary);">📂 ${source}</li>`;
        });
        html += "</ul></div>";
    }
    
    return html;
}

function parseTablesInText(text) {
    const lines = text.split("\n");
    let inTable = false;
    let tableLines = [];
    let processedLines = [];

    for (let line of lines) {
        const isTableRow = line.trim().startsWith("|") && line.trim().endsWith("|");
        if (isTableRow) {
            inTable = true;
            tableLines.push(line);
        } else {
            if (inTable) {
                processedLines.push(compileTableHtml(tableLines));
                tableLines = [];
                inTable = false;
            }
            processedLines.push(line);
        }
    }
    if (inTable) {
        processedLines.push(compileTableHtml(tableLines));
    }
    return processedLines.join("\n");
}

function compileTableHtml(rows) {
    if (rows.length === 0) return "";
    
    let tableHtml = '<table style="width:100%; border-collapse:collapse; margin:12px 0; border:1px solid rgba(255,255,255,0.05);">';
    let bodyRows = [...rows];
    
    let hasHeader = false;
    if (rows.length > 1 && rows[1].includes("-")) {
        hasHeader = true;
    }
    
    if (hasHeader) {
        const headerCells = rows[0].split("|").slice(1, -1).map(c => c.trim());
        tableHtml += "<thead><tr style=\"background:rgba(255,255,255,0.02);\">";
        headerCells.forEach(cell => {
            tableHtml += `<th style="padding:10px 14px; border:1px solid rgba(255,255,255,0.05); text-align:left; font-size:13px; font-weight:600;">${escapeHtml(cell)}</th>`;
        });
        tableHtml += "</tr></thead>";
        bodyRows = rows.slice(2);
    }
    
    tableHtml += "<tbody>";
    bodyRows.forEach(row => {
        const cells = row.split("|").slice(1, -1).map(c => c.trim());
        tableHtml += "<tr>";
        cells.forEach(cell => {
            tableHtml += `<td style="padding:10px 14px; border:1px solid rgba(255,255,255,0.05); font-size:13px; color:var(--text-secondary);">${escapeHtml(cell)}</td>`;
        });
        tableHtml += "</tr>";
    });
    tableHtml += "</tbody></table>";
    
    return tableHtml;
}

let activeTypewriter = null;

function typeHtml(element, htmlContent, speed, onComplete) {
    let currentHtml = "";
    let isTag = false;
    let i = 0;
    let cancelled = false;

    if (activeTypewriter) {
        activeTypewriter.cancel();
    }

    activeTypewriter = {
        cancel: () => {
            cancelled = true;
        }
    };
    
    function step() {
        if (cancelled) {
            activeTypewriter = null;
            if (onComplete) onComplete();
            return;
        }
        if (i >= htmlContent.length) {
            activeTypewriter = null;
            if (onComplete) onComplete();
            return;
        }
        
        let char = htmlContent[i];
        
        if (char === '<') {
            isTag = true;
        }
        
        currentHtml += char;
        
        if (char === '>') {
            isTag = false;
        }
        
        element.innerHTML = currentHtml;
        scrollToBottom();
        
        i++;
        
        if (isTag) {
            step();
        } else {
            setTimeout(step, speed);
        }
    }
    
    step();
}

function showStopButton() {
    const sendBtn = document.getElementById("send-message-btn");
    const stopBtn = document.getElementById("stop-generation-btn");
    if (sendBtn && stopBtn) {
        sendBtn.style.display = "none";
        stopBtn.style.display = "flex";
    }
}

function hideStopButton() {
    const sendBtn = document.getElementById("send-message-btn");
    const stopBtn = document.getElementById("stop-generation-btn");
    if (sendBtn && stopBtn) {
        sendBtn.style.display = "flex";
        stopBtn.style.display = "none";
    }
}

window.stopGeneration = function() {
    if (activeTypewriter) {
        activeTypewriter.cancel();
    }
    hideStopButton();
};
