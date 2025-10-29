FROM alpine:latest
RUN apk add --no-cache git
RUN git config --global --add safe.directory /workspace
RUN git config --global user.name "Bot" && git config --global user.email "bot@github.com"
WORKDIR /workspace
COPY . /workspace/
RUN chown -R root:root /workspace
ENV GITHUB_USER=Huxufeng
ENV GITHUB_TOKEN=
ENV GITHUB_REPO=Multi-Scale
CMD sh -c "rm -rf .git && git init && git add . && git commit -m 'upload' && git branch -M main && git remote add origin https://\$GITHUB_USER:\$GITHUB_TOKEN@github.com/\$GITHUB_USER/\$GITHUB_REPO.git || true && git push -u origin main --force"
