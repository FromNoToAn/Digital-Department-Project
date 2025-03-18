<script setup>
import { ref } from "vue";

const selectedFile = ref(null);
const status = ref("");
const progress = ref(0);
const video_url = ref("");
const preview_url = ref("");
const site = ref("http://127.0.0.1:8000");

const load_flag = ref(true);

let checkInterval = null;

const handleFileChange = (event) => {
  selectedFile.value = event.target.files[0];
};

const uploadVideo = async () => {
  if (!selectedFile.value) return;

  const formData = new FormData();
  formData.append("video", selectedFile.value);

  try
  {
    const response = await fetch(`${site.value}/upload`, {
      method: "POST",
      body: formData,
    });

    console.log("Статус ответа:", response.status);

    if (!response.ok) throw new Error("Ошибка загрузки видео");

    const data = await response.json();
    status.value = `Видео загружено. task_id: ${data.task_id}`;

    preview_url.value = `${site.value}/task/status/${data.task_id}`;
    video_url.value = `${site.value}/${data.video_url}`;

    checkInterval = setInterval(() => checkVideoStatus(data.task_id), 1500);

    console.log(preview_url.value);
    console.log(video_url.value);
  }
  catch (error)
  {
    console.error("Ошибка:", error);
    status.value = "Ошибка при загрузке видео";
  }
};

const checkVideoStatus = async (task_id) => {
  try
  {
    const response = await fetch(`${site.value}/task/status/${task_id}`, { method: "GET" });
    load_flag.value = false;

    if (response.ok)
    {
      const data = await response.json();
      if(data.progress)
      {
        progress.value = data.progress;
      }
      
      // Проверяем поле success в ответе
      if (data.success)
      {
        clearInterval(checkInterval);
        preview_url.value = "";
        video_url.value = `${site.value}/videos/${task_id}.mp4`;
      }
      else
      {
        console.log("Видео еще не готово...");
      }
    }
    else
    {
      console.error("Ошибка запроса к серверу...");
    }
  }
  catch (error)
  {
    console.error("Ошибка при проверке статуса видео:", error);
  }
};
</script>

<template>
  <section class="section upload_section">
    <div class="section_container">
      <div class="section_header">
        <div class="text">Загрузка видео на сервер</div>
      </div>
      <div class="section_text">
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
      </div>
    </div>
  </section>
  <section v-if="status && (preview_url || video_url)" class="section upload_section">
    <div class="section_container">
      <div class="section_header">
        <div class="text">Обработанное видео</div>
      </div>
      <div class="section_text for_video">
        <div class="section_progress" v-if="preview_url" :src="video_url">
          <div class="progress_bar-container">
            <div class="progress_bar" :style="{ width: progress * 100 + '%' }"></div>
          </div>
          <div class="progress_prersentage">{{ (progress* 100).toFixed(2) }}%</div>
        </div>

        <div class="section_video" v-if="!load_flag && !preview_url">
          <video class="result_video" controls>
            <source :src="video_url" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    </div>
  </section>
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
</style>
