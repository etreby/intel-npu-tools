// CI/CD pipeline for intel-npu-tools
// Stages: pytest -> build .deb -> publish to Nexus -> push image to Harbor -> deploy .deb to workstation
pipeline {
    agent any

    environment {
        HARBOR_REGISTRY = "harbor.home.hotelsarena.com"
        HARBOR_PROJECT = "etreby"
        IMAGE_NAME = "intel-npu-tools"
        IMAGE_TAG = "${BUILD_NUMBER}-${GIT_COMMIT.take(8)}"
        FULL_IMAGE = "${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}"
        NEXUS_URL = "http://192.168.1.10:30451/repository/raw-builds/intel-npu-tools"
        WORKSTATION = "etreby@192.168.1.31"
    }

    stages {
        stage("Prepare") {
            steps {
                // The agent talks to the host docker daemon via the mounted socket:
                // bind-mount sources must be valid paths ON THE HOST, not inside the
                // Jenkins container. JENKINS_HOME is mounted from
                // /mnt/BigData/Apps/jenkins/home, so translate $PWD accordingly.
                script {
                    env.HOST_WORKSPACE = sh(returnStdout: true,
                        script: 'echo /mnt/BigData/Apps/jenkins/home${PWD#/var/jenkins_home}').trim()
                    echo "Host workspace: ${env.HOST_WORKSPACE}"
                }
            }
        }

        stage("Test (pytest)") {
            steps {
                // Build the release image once, then run the pytest suite inside it
                sh '''
                    docker build -q -t ${FULL_IMAGE}:${IMAGE_TAG} .
                    docker run --rm -v "${HOST_WORKSPACE}:/src" -w /src ${FULL_IMAGE}:${IMAGE_TAG} \
                        sh -c "pip install -q pytest && pytest tests/ -q"
                '''
            }
        }

        stage("Build Debian Package (.deb)") {
            steps {
                // Hermetic build in debian:bookworm-slim (librsvg2-bin for icon conversion, fakeroot for dpkg-deb)
                sh '''
                    docker run --rm -v "${HOST_WORKSPACE}:/src" -w /src debian:bookworm-slim \
                        bash -c "apt-get update -qq && apt-get install -y -qq --no-install-recommends librsvg2-bin fakeroot >/dev/null && bash scripts/build-deb.sh dist/"
                    chmod 644 dist/*.deb
                    ls -la dist/
                '''
            }
        }

        stage("Stage Artifact in Nexus") {
            steps {
                withCredentials([usernamePassword(credentialsId: "nexus-credentials",
                                                  usernameVariable: "NEXUS_USER",
                                                  passwordVariable: "NEXUS_PASS")]) {
                    sh '''
                        DEB_FILE=$(ls dist/*.deb | head -1)
                        DEB_BASENAME=$(basename "$DEB_FILE")
                        sha256sum "$DEB_FILE" > "$DEB_FILE.sha256"

                        echo "Uploading $DEB_BASENAME to Nexus Artifactory..."
                        curl -sf -u "${NEXUS_USER}:${NEXUS_PASS}" --upload-file "$DEB_FILE" "${NEXUS_URL}/${DEB_BASENAME}"
                        curl -sf -u "${NEXUS_USER}:${NEXUS_PASS}" --upload-file "$DEB_FILE.sha256" "${NEXUS_URL}/${DEB_BASENAME}.sha256"
                        echo "Artifact staged in Nexus: ${NEXUS_URL}/${DEB_BASENAME}"
                    '''
                }
            }
        }

        stage("Build & Push Harbor Container") {
            steps {
                withCredentials([usernamePassword(credentialsId: "harbor-credentials",
                                                  usernameVariable: "HARBOR_USER",
                                                  passwordVariable: "HARBOR_PASS")]) {
                    sh '''
                        echo "${HARBOR_PASS}" | docker login ${HARBOR_REGISTRY} -u "${HARBOR_USER}" --password-stdin
                        docker tag ${FULL_IMAGE}:${IMAGE_TAG} ${FULL_IMAGE}:latest
                        docker push ${FULL_IMAGE}:${IMAGE_TAG}
                        docker push ${FULL_IMAGE}:latest
                        docker logout ${HARBOR_REGISTRY}
                        echo "Container image pushed to Harbor: ${FULL_IMAGE}:${IMAGE_TAG}"
                    '''
                }
            }
        }

        stage("Deploy to Workstation") {
            steps {
                withCredentials([sshUserPrivateKey(credentialsId: "workstation-deploy-key",
                                                   keyFileVariable: "WS_KEY")]) {
                    sh '''
                        DEB_FILE=$(ls dist/*.deb | head -1)
                        DEB_BASENAME=$(basename "$DEB_FILE")

                        echo "Deploying $DEB_BASENAME to workstation (${WORKSTATION})..."
                        scp -i "$WS_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
                            "$DEB_FILE" ${WORKSTATION}:/tmp/${DEB_BASENAME}

                        ssh -i "$WS_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new ${WORKSTATION} \
                            "sudo dpkg -i /tmp/${DEB_BASENAME} || { sudo apt-get install -f -y && sudo dpkg -i /tmp/${DEB_BASENAME}; }"

                        ssh -i "$WS_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new ${WORKSTATION} \
                            "dpkg-query -W -f='\\${Status} \\${Version}\\n' intel-npu-tools"
                        echo "Deployment verified on workstation."
                    '''
                }
            }
        }
    }

    post {
        success {
            echo "intel-npu-tools CI/CD pipeline succeeded (Build #${BUILD_NUMBER})."
        }
        failure {
            echo "intel-npu-tools CI/CD pipeline failed (Build #${BUILD_NUMBER})."
        }
    }
}
