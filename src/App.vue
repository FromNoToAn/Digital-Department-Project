<script setup>
  import VideoUploader from "./components/VideoUploader.vue";
</script>

<template>
  <header class="header">
    <img class="picture" src="/images/animals.svg" alt="Logo">
    <div class="text">Animal <span class="main-gradient">Detection</span></div>
  </header>
  <section class="sections">
    <VideoUploader />
  </section>
</template>

<style scoped lang="scss">
@use '../src/styles/main.scss' as global;
</style>

