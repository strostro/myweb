#!/bin/bash

set -e

echo "📝 Step 0: Commit updates to main..."

git add .
git commit -m "chore: update site content" || echo "✅ No changes to commit"
git push origin main

echo "🚀 Step 1: Build the site with Jekyll..."
JEKYLL_ENV=production bundle exec jekyll build

echo "📁 Step 2: Deploy _site to gh-pages branch..."

cd _site

git init
git remote remove origin 2> /dev/null
git remote add origin https://github.com/strostro/myweb.git
git branch -D gh-pages 2> /dev/null      # ✅ 删除已有的 gh-pages 分支（如果有）
git checkout -b gh-pages

git add .
git commit -m "deploy: publish site"
git push -f origin gh-pages

cd ..
echo "🎉 Deployed to GitHub Pages!"