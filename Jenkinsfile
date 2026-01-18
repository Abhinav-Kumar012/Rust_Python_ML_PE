pipeline {
    agent any

    environment {
        IMAGE_NAME = "mnist-infer"
        METRICS_FILE = "build_metrics.json"
    }

    triggers {
        // Trigger on changes to specific folders, if supported by SCM plugin
        // Otherwise, this logic is often handled by the webhook configuration or
        // by checking changesets in the pipeline steps.
        // For this example, we assume the webhook triggers the build.
        pollSCM('') 
    }

    stages {
        stage('Check Changes') {
            steps {
                script {
                    // Simple check if relevant folders changed (git diff against previous commit)
                    // This is a basic approximation.
                    def changed = sh(script: "git diff --name-only HEAD HEAD~1 | grep -E 'rust_ml/mnist_ml|rust_ml/mnist_infer'", returnStatus: true) == 0
                    if (!changed) {
                        echo "No changes in relevant folders. Skipping build."
                        // In a real pipeline, we might mark as UNSTABLE or abort, 
                        // but strictly stopping valid stages is complex in declarative without plugins.
                        // We will proceed for demonstration purposes or assume the webhook filters.
                    }
                }
            }
        }

        stage('Build & Test') {
            steps {
                script {
                    def buildStart = System.currentTimeMillis()
                    try {
                        // Build Docker Image from Repo Root context
                         sh "docker build -t ${IMAGE_NAME} -f rust_ml/mnist_infer/Dockerfile ."
                        
                        def buildDuration = System.currentTimeMillis() - buildStart
                        currentBuild.description = "Build Success (${buildDuration}ms)"
                        
                        // 1. CI/CD Build Behavior (captured in metrics)
                        writeJSON(file: 'build_behavior.json', json: [
                            status: 'SUCCESS',
                            duration_ms: buildDuration,
                            timestamp: new Date().format("yyyy-MM-dd'T'HH:mm:ss'Z'")
                        ])

                    } catch (Exception e) {
                        def buildDuration = System.currentTimeMillis() - buildStart
                        writeJSON(file: 'build_behavior.json', json: [
                            status: 'FAILURE',
                            duration_ms: buildDuration,
                            error: e.getMessage()
                        ])
                        error("Build failed: " + e.getMessage())
                    }
                }
            }
        }

        stage('Measure Metrics') {
            steps {
                script {
                    // 2. Container Image Size & Layering
                    def imageSize = sh(script: "docker inspect -f '{{ .Size }}' ${IMAGE_NAME}", returnStdout: true).trim()
                    def layerCount = sh(script: "docker history -q ${IMAGE_NAME} | wc -l", returnStdout: true).trim()

                    echo "Image Size: ${imageSize} bytes"
                    echo "Layer Count: ${layerCount}"

                    // 3. Cold Start Latency
                    // Using a helper script or inline shell
                    // Run container detached
                    sh "docker run -d -p 8080:8080 --name ${IMAGE_NAME}-test ${IMAGE_NAME}"
                    
                    // Wait for it to be ready and measure time
                    // We'll use a simple loop in shell
                    def coldStartScript = '''
                        start=$(date +%s%3N)
                        for i in {1..30}; do
                            if curl -s http://localhost:8080/health > /dev/null; then
                                end=$(date +%s%3N)
                                echo $((end-start))
                                exit 0
                            fi
                            sleep 0.5
                        done
                        echo -1
                    '''
                    def coldStartLatency = sh(script: coldStartScript, returnStdout: true).trim()
                    
                    echo "Cold Start Latency: ${coldStartLatency} ms"

                    // Cleanup
                    sh "docker rm -f ${IMAGE_NAME}-test"

                    // Persist Metrics
                    def metrics = [
                        build: readJSON(file: 'build_behavior.json'),
                        image: [
                            size_bytes: imageSize.toLong(),
                            layers: layerCount.toInteger()
                        ],
                        runtime: [
                            cold_start_latency_ms: coldStartLatency.toInteger()
                        ]
                    ]

                    writeJSON(file: METRICS_FILE, json: metrics, pretty: 4)
                    archiveArtifacts artifacts: METRICS_FILE
                }
            }
        }
    }
}
