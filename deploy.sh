#!/bin/bash

set -e  # Exit on any error

echo "🚀 Step 1: Push content to main branch..."
git add .
git commit -m "✨ Update blog content"
git push origin main

echo "✅ Main branch updated."

echo "🔨 Step 2: Build the site using Jekyll..."
JEKYLL_ENV=production bundle exec jekyll build

echo "📁 Step 3: Deploy _site to gh-pages branch..."
cd _site
git init
git remote add origin https://github.com/strostro/myweb.git
git checkout -b gh-pages
git add .
git commit -m "🚀 Deploy site"
git push -f origin gh-pages

echo "🎉 Done! Your website is live!"