module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      ['react-native-worklets-core/plugin', { processNestedWorklets: true }],
      ['react-native-reanimated/plugin', { relativeSourceLocation: true }],
    ],
  };
};
