<script setup>
  import VideoUploader from "./components/VideoUploader.vue";
</script>

<template>
  <header class="header">
    <div class="text">Animal <span class="main-gradient">Detection</span></div>
  </header>
  <section class="section upload_section">
    <div class="section_container">
      <div class="section_header">
        <div class="text">Загрузка видео на сервер</div>
      </div>
      <div class="section_text">
        <VideoUploader />
      </div>
    </div>
  </section>
</template>

<style scoped lang="scss">
@use '../src/styles/main.scss' as global;
</style>

