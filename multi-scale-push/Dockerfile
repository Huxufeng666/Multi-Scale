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
