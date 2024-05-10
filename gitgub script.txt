#!/bin/bash

# إدخال البيانات الأساسية
read -p "Diagnosing Colorectal and Lung Cancer from Histopatho-logical Images using Machine Learning Approach" PROJECT_FOLDER
read -p "full project has been added" COMMIT_MESSAGE
read -p "تاريخ الكوميت (مثال: 2024-05-10T10:00:00): " COMMIT_DATE
read -p "Hamzah A.A Amir" GIT_NAME
read -p "alhamza2012@gmail.com" GIT_EMAIL
read -p "https://github.com/HamzaAmir97/Diagnosing-Colorectal-and-Lung-Cancer-from-Histopatho-logical-Images-using-Machine-Learning-Approach.git" GITHUB_REPO

# التنقل داخل مجلد المشروع
cd "$PROJECT_FOLDER" || { echo "المجلد غير موجود!"; exit 1; }

# تهيئة Git
Git init
git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"

# إضافة الملفات
git add .

# إنشاء الكوميت بتاريخ قديم
GIT_AUTHOR_DATE="$COMMIT_DATE" GIT_COMMITTER_DATE="$COMMIT_DATE" git commit -m "$COMMIT_MESSAGE"

# إنشاء الفرع الرئيسي وربطه بـ GitHub
git branch -M main
git remote add origin "$GITHUB_REPO"
git push -u origin main

echo "تم رفع المشروع بنجاح بتاريخ $COMMIT_DATE!"

