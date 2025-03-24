<script setup>
import { ref } from "vue";

const site = ref(import.meta.env.VITE_API_BASE_URL);

const selected_file = ref(null);
const is_realtime = ref(false);

const is_realtime_flag = ref(false);
const process_flag= ref(false);
const load_flag = ref(true);

const info = ref("");
const progress = ref(0);

const video_url = ref("");
const img_url = ref("/images/no_img.jpg");

const processing_header = ref("Обработка видео...");

let checkInterval = null;
let streamInterval = null;

const handleFileChange = (event) => {
  selected_file.value = event.target.files[0];
};

const uploadVideo = async () => {
  if (!selected_file.value) return;

  is_realtime_flag.value = is_realtime.value;

  const formData = new FormData();
  formData.append("video", selected_file.value);
  formData.append("is_realtime", is_realtime_flag.value);

  try
  {
    const response = await fetch(`${site.value}/upload_video`, {
      method: "POST",
      body: formData,
    });

    console.log("Статус ответа:", response.status);

    if (!response.ok) throw new Error("Ошибка загрузки видео");

    const data = await response.json();
    info.value = `Видео загружено. task_id: ${data.task_id}`;

    process_flag.value = true;

    checkInterval = setInterval(() => checkVideoStatus(data.task_id), 1500);

    if (is_realtime_flag.value)
    {
      streamInterval = setInterval(() => checkStreamStatus(data.task_id), 100);
    }

  }
  catch (error)
  {
    console.error("Ошибка:", error);
    info.value = "Ошибка при загрузке видео";
  }
};

const checkVideoStatus = async (task_id) => {
  try
  {
    const response_status = await fetch(`${site.value}/task/status/${task_id}`, { method: "GET" });
    load_flag.value = false;

    if (response_status.ok)
    {
      const data = await response_status.json();
      if(data.progress && !is_realtime_flag.value)
      {
        progress.value = data.progress;
      }
      
      if (data.mp4)
      {
        clearInterval(checkInterval);
        process_flag.value = false;
        video_url.value = `${site.value}/videos/${task_id}.mp4?t=${Date.now()}`;
        processing_header.value = "Обработанное видео";
        console.log(video_url.value);
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

const checkStreamStatus = async (task_id) => {
  try
  {
    const response = await fetch(`${site.value}/task/stream/${task_id}`);
    const data = await response.json();

    // console.log("Stream статус:", data);

    if (data.stream && process_flag.value)
    {
      if (data.file_url)
      {
        const new_img_url = `${data.file_url}?t=${Date.now()}`;
        const img = new Image();
        img.src = new_img_url;

        img.onload = () => {
          img_url.value = new_img_url;
        };

        img.onerror = () => {
          console.warn("Ошибка загрузки кадра, оставляем старый URL");
        };
      }
    }
    else
    {
      clearInterval(streamInterval);
    }
  }
  catch (error)
  {
    console.error("Ошибка при получении потока:", error);
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
          <div class="upload_section">
            <div class="checkbox_container">
              <input type="checkbox" id="realtime" v-model="is_realtime" class="hidden_checkbox"/>
              <label for="realtime" class="custom_checkbox">
                <div class="checkmark"></div>
                <div class="text">Реалтайм</div>
              </label>
            </div>
            <div class="file_input">
              <label for="fileInput" class="select_button">Выбрать<br/>видео</label>
              <input class="select_input" id="fileInput" type="file" accept="video/mp4" @change="handleFileChange" hidden/>
              <p class="select_file" v-if="selected_file">{{ selected_file.name }}</p>
            </div>
          </div>

          <div class="file_send" v-if="selected_file">
            <button class="send_button" @click="uploadVideo" :disabled="!selected_file">Загрузить видео</button>
            <p class="select_file" v-if="info">{{ info }}</p>
          </div>
        </div>
      </div>
    </div>
  </section>
  <section v-if="info && (process_flag || video_url)" class="section upload_section">
    <div class="section_container">
      <div class="section_header">
        <div class="text">{{ processing_header }}</div>
      </div>
      <div class="section_text for_video">
        <div class="section_progress" v-if="process_flag && !is_realtime_flag" :src="video_url">
          <div class="progress_bar-container">
            <div class="progress_bar" :style="{ width: progress * 100 + '%' }"></div>
          </div>
          <div class="progress_prersentage">{{ (progress* 100).toFixed(2) }}%</div>
        </div>

        <div class="section_img" v-if="process_flag && is_realtime_flag">
          <img class="progress_img" :src="img_url" alt="Потоковое изображение"/>
        </div>

        <div class="section_video" v-if="!process_flag && !load_flag">
          <video class="result_video" controls>
            <source :src="video_url" type="video/mp4"/>
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
  gap: calc(var(--mini-margin));

  @media(max-width: 550px)
  {
    flex-direction: column;
  }
}

.upload_section
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
  @include global.button(var(--color-light_lime));
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


.checkbox_container
{
  display: flex;
  flex-direction: row;
  align-items: center;

  font-family: var(--font-muller-medium);
  font-size: var(--font-size);
}

/* Скрываем стандартный чекбокс */
.hidden_checkbox
{
  display: none;
}

/* Стиль кастомного чекбокса */
.custom_checkbox
{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: calc(var(--mini-margin) / 4);
  user-select: none;
  cursor: pointer;
}

/* Внешний квадрат */
.checkmark
{
  display: flex;
  align-items: center;
  justify-content: center;
  width: calc(var(--font-size) * 1.8);
  aspect-ratio: 1/1;
  border: calc(var(--border-size) / 2) solid var(--color-dark);
  border-radius: calc(var(--border-radius) / 4);
  background-color: var(--color-light);
  background-color: var(--color-white);
  transition: 0.25s ease-in-out;
}

.custom_checkbox:hover .checkmark
{
  border-color: var(--color-lime);
}

.checkmark::after
{
  content: "✔";
  color: var(--color-white);
  display: none;
}

.hidden_checkbox:checked + .custom_checkbox .checkmark
{
  border-color: var(--color-lime);
  background-color: var(--color-dark);
}

.hidden_checkbox:checked + .custom_checkbox .checkmark::after
{
  display: block;
}
</style>
