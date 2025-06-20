// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// .tflite 파일 확장자 추가
config.resolver.assetExts.push('tflite');

module.exports = config;
