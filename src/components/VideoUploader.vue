<script setup>
import { ref } from "vue";

const selectedFile = ref(null);
const status = ref("");

const handleFileChange = (event) => {
  selectedFile.value = event.target.files[0];
};

const uploadVideo = async () => {
  if (!selectedFile.value) return;

  const formData = new FormData();
  formData.append("video", selectedFile.value);

  try
  {
    const response = await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    });

    console.log("Статус ответа:", response.status);

    if (!response.ok) throw new Error("Ошибка загрузки видео");

    const data = await response.json();
    status.value = `Видео загружено. task_id: ${data.task_id}`;
  }
  catch (error)
  {
    console.error("Ошибка:", error);
    status.value = "Ошибка при загрузке видео";
  }
};
</script>

<template>
  <div class="uploader">
    <div class="file_input">
      <label for="fileInput" class="select_button">Выбрать<br/>видео</label>
      <input class="select_input" id="fileInput" type="file" accept="video/mp4" @change="handleFileChange" hidden/>
      <p class="select_file" v-if="selectedFile">{{ selectedFile.name }}</p>
    </div>
    <div class="file_send">
      <button class="send_button" @click="uploadVideo" :disabled="!selectedFile">Загрузить видео</button>
      <p class="select_file" v-if="status">{{ status }}</p>
    </div>
  </div>
</template>

<style scoped lang="scss">
@use '../../src/styles/main.scss' as global;

.uploader
{
  position: relative;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: calc(var(--mini-margin));
}

.file_input,
.file_send
{
  position: relative;
  display: flex;
  flex-direction: row;
  align-items: center;
}

.select_button
{
  @include global.button(var(--color-black));
  background-color: var(--color-dark);
  cursor: pointer;
}

.select_file
{
  position: relative;
  display: flex;
  padding: calc(var(--mini-margin) / 4);
  justify-content: center;
  align-items: center;
  text-align: center;
  font-family: var(--font-muller-medium);
  font-size: var(--font-size);
}

.send_button
{
  @include global.button(var(--color-green));
  border: none;
  color: var(--color-white);
  background-color: var(--color-lime);
  cursor: pointer;
}

.send_button:disabled
{
  display: none;
  cursor: not-allowed;
}

// button
// {
//   padding: 10px;
//   cursor: pointer;
//   background-color: #4CAF50;
//   color: white;
//   border: none;
//   border-radius: 5px;
// }
</style>
