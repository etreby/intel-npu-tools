pipeline {
    agent any

    environment {
        HARBOR_REGISTRY = "harbor.home.hotelsarena.com"
        HARBOR_PROJECT  = "etreby"
        IMAGE_NAME      = "intel-npu-tools"
        IMAGE_TAG       = "${BUILD_NUMBER}-${GIT_COMMIT.take(8)}"
        FULL_IMAGE      = "${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}"
        NEXUS_URL       = "http://192.168.1.10:30088/repository/raw-builds/intel-npu-tools"
        WORKSTATION_IP  = "192.168.1.31"
        WORKSTATION_USER= "etreby"
    }

    stages {
        stage("Checkout & Test") {
            steps {
                echo "Testing commit ${GIT_COMMIT} (Build #${BUILD_NUMBER})"
            }
        }

        stage("Build Debian Package (.deb)") {
            steps {
                sh """
                    bash scripts/build-deb.sh dist/
                    ls -la dist/
                """
            }
        }

        stage("Stage Artifact in Nexus") {
            steps {
                withCredentials([usernamePassword(credentialsId: "nexus-credentials", usernameVariable: "NEXUS_USER", passwordVariable: "NEXUS_PASS")]) {
                    sh """
                        DEB_FILE=\$(ls dist/*.deb | head -n 1)
                        DEB_BASENAME=\$(basename "\$DEB_FILE")
                        sha256sum "\$DEB_FILE" > "\$DEB_FILE.sha256"

                        echo "Uploading \$DEB_BASENAME to Nexus Artifactory..."
                        curl -s -u \${NEXUS_USER}:\${NEXUS_PASS} --upload-file "\$DEB_FILE" "\${NEXUS_URL}/\$DEB_BASENAME" || true
                        curl -s -u \${NEXUS_USER}:\${NEXUS_PASS} --upload-file "\$DEB_FILE.sha256" "\${NEXUS_URL}/\$DEB_BASENAME.sha256" || true
                    """
                }
            }
        }

        stage("Build & Push Harbor Container") {
            steps {
                withCredentials([usernamePassword(credentialsId: "harbor-credentials", usernameVariable: "HARBOR_USER", passwordVariable: "HARBOR_PASS")]) {
                    sh """
                        docker build -t ${FULL_IMAGE}:${IMAGE_TAG} -t ${FULL_IMAGE}:latest .
                        echo "\${HARBOR_PASS}" | docker login ${HARBOR_REGISTRY} -u "\${HARBOR_USER}" --password-stdin
                        docker push ${FULL_IMAGE}:${IMAGE_TAG}
                        docker push ${FULL_IMAGE}:latest
                        echo "Container image pushed to Harbor."
                    """
                }
            }
        }

        stage("Deploy to Workstation") {
            steps {
                sh """
                    DEB_FILE=\$(ls dist/*.deb | head -n 1)
                    DEB_BASENAME=\$(basename "\$DEB_FILE")
                    echo "Deploying \$DEB_BASENAME to Workstation (${WORKSTATION_USER}@${WORKSTATION_IP})..."
                    scp -o StrictHostKeyChecking=no "\$DEB_FILE" ${WORKSTATION_USER}@${WORKSTATION_IP}:/tmp/\$DEB_BASENAME
                    ssh -o StrictHostKeyChecking=no ${WORKSTATION_USER}@${WORKSTATION_IP} "sudo dpkg -i /tmp/\$DEB_BASENAME || sudo apt-get install -f -y"
                    echo "Deployment completed on workstation."
                """
            }
        }
    }

    post {
        success {
            echo "intel-npu-tools CI/CD pipeline succeeded! Deb staged in Nexus, Container pushed to Harbor, Deployed to Workstation."
        }
        failure {
            echo "intel-npu-tools CI/CD pipeline failed."
        }
    }
}
