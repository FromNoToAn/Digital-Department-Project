<script setup>
  import VideoUploader from "./components/VideoUploader.vue";
</script>

<template>
  <div id="app">
    <h1>Загрузка видео на сервер FastAPI</h1>
    <VideoUploader />
  </div>
</template>

<style>
#app {
  max-width: 600px;
  margin: auto;
  text-align: center;
  font-family: Arial, sans-serif;
}
h1 {
  font-size: 24px;
  margin-bottom: 20px;
}
</style>

