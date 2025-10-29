#!/bin/bash
set -e

# ============ 配置 ============
GITHUB_USER="Huxufeng666"
GITHUB_REPO="Multi-Scale"
PROJECT_DIR="multi-scale-push"
# ==============================

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${GREEN}[INFO]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 检查环境
[ -f /.dockerenv ] && error "请在宿主机运行！"
docker info >/dev/null 2>&1 || { log "启动 Docker..."; sudo systemctl start docker; sleep 2; }
[ -z "$GITHUB_TOKEN" ] && error "请设置 GITHUB_TOKEN！\n用法: GITHUB_TOKEN=ghp_xxx ./run.sh"

# 创建项目
log "创建项目目录: $PROJECT_DIR"
rm -rf "$PROJECT_DIR" && mkdir -p "$PROJECT_DIR" && cd "$PROJECT_DIR"

# 创建 README
log "创建 README.md"
echo "# Multi-Scale" > README.md

# .dockerignore（关键！）
log "创建 .dockerignore"
cat > .dockerignore << 'EOF'
.git
Dockerfile
.dockerignore
EOF

# Dockerfile（关键修复！）
log "创建 Dockerfile"
cat > Dockerfile << 'EOF'
FROM alpine:latest

# 1. 安装 git
RUN apk add --no-cache git

# 2. 【关键】先设置 safe.directory（在 COPY 之前！）
RUN git config --global --add safe.directory /workspace

# 3. 设置 git 用户
RUN git config --global user.name "Docker" && \
    git config --global user.email "docker@github.com"

# 4. 工作目录
WORKDIR /workspace

# 5. 复制代码（.git 被 .dockerignore 排除）
COPY . /workspace/

# 6. 修复所有权
RUN chown -R root:root /workspace

# 7. 执行命令
CMD ["/bin/sh", "-c", "\
    rm -rf .git && \
    git init && \
    echo '# Multi-Scale' > README.md && \
    git add . && \
    git commit -m 'first commit' && \
    git branch -M main && \
    git remote add origin https://$GITHUB_USER:$GITHUB_TOKEN@github.com/$GITHUB_USER/$GITHUB_REPO.git || true && \
    git push -u origin main --force && \
    echo '上传成功！' \
"]
EOF

# 构建 + 运行
log "构建镜像..."
docker build -t push-ms .

log "上传到 GitHub..."
docker run --rm \
  -e GITHUB_USER="$GITHUB_USER" \
  -e GITHUB_TOKEN="$GITHUB_TOKEN" \
  -e GITHUB_REPO="$GITHUB_REPO" \
  push-ms

echo "成功！查看: https://github.com/$GITHUB_USER/$GITHUB_REPO"