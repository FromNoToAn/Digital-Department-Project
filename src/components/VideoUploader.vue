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
    <input type="file" accept="video/mp4" @change="handleFileChange" />
    <button @click="uploadVideo" :disabled="!selectedFile">Загрузить видео</button>
    <p v-if="status">{{ status }}</p>
  </div>
</template>

<style scoped>
.uploader {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: center;
}
button {
  padding: 10px;
  cursor: pointer;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 5px;
}
button:disabled {
  background-color: #ddd;
  cursor: not-allowed;
}
</style>
