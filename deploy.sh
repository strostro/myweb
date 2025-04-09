#!/bin/bash

set -e  # 遇到错误就终止脚本

echo "📝 Step 0: Commit updates to main..."

git add .
git commit -m "chore: update site content" || echo "No changes to commit"
git push origin main

echo "🚀 Step 1: Build the site with Jekyll..."
JEKYLL_ENV=production bundle exec jekyll build

echo "📁 Step 2: Deploy _site to gh-pages branch..."

cd _site

# 初始化并配置 git 仓库
git init
git remote remove origin 2> /dev/null  # 先删除已有的 origin（如果有）
git remote add origin https://github.com/strostro/myweb.git

git checkout -b gh-pages

git add .
git commit -m "deploy: publish site"

# 强制推送到 gh-pages 分支
git push -f origin gh-pages

cd ..
echo "🎉 Deployed to GitHub Pages!"