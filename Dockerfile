# Dockerfile
FROM alpine:latest

# 安装 git
RUN apk add --no-cache git

# 关键：先设置 safe.directory（在 COPY 之前！）
RUN git config --global --add safe.directory /workspace

# 设置 git 用户（避免提交警告）
RUN git config --global user.name "hxf" && \
    git config --global user.email "hxufeng66@gmail.com"

# 工作目录
WORKDIR /workspace

# 复制代码（.git 被 .dockerignore 排除）
COPY . /workspace/

# 修复所有权（让 root 完全拥有）
RUN chown -R root:root /workspace && \
    find /workspace -type d -exec chmod 755 {} \; && \
    find /workspace -type f -exec chmod 644 {} \;

# 环境变量
ENV GITHUB_USER=Huxufeng666
ENV GITHUB_TOKEN=
ENV GITHUB_REPO=Multi-Scale
ENV GITHUB_BRANCH=main

# 自动执行（干净初始化 + 推送）
CMD sh -c "\
    echo '=== 清理旧 .git ===' && \
    rm -rf .git && \
    echo '=== git init ===' && \
    git init && \
    echo '=== git add ===' && \
    git add . && \
    echo '=== git commit ===' && \
    git commit -m 'Auto upload from Docker' || echo 'No changes' && \
    echo '=== git branch ===' && \
    git branch -M $GITHUB_BRANCH && \
    echo '=== git remote ===' && \
    git remote add origin https://$GITHUB_USER:$GITHUB_TOKEN@github.com/$GITHUB_USER/$GITHUB_REPO.git || true && \
    echo '=== git push ===' && \
    git push -u origin $GITHUB_BRANCH --force && \
    echo '上传成功！' \
"