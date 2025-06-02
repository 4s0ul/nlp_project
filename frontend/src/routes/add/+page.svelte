<!-- src/routes/add/+page.svelte -->
<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';

  // Получаем id (если пришёл ?id=...)
  let id;
  $: id = $page.url.searchParams.get('id');

  // Список тем
  let topics = [];
  let loadingTopics = false;
  let errorTopics = null;

  // Поля формы
  let topic_id = '';
  let raw_text = '';
  let description_raw_text = '';
  let language = 'russian';
  let info = '';

  // Состояние загрузки слова при редактировании
  let loadingWord = false;
  let errorWord = null;

  // Состояние сабмита
  let submitting = false;
  let submitError = null;

  // 1) Загрузить список тем
  async function loadTopics() {
    loadingTopics = true;
    errorTopics = null;
    try {
      const res = await fetch('http://localhost:8000/topics/', {
        headers: { Accept: 'application/json' }
      });
      if (!res.ok) throw new Error(`Ошибка загрузки тем: ${res.status}`);
      topics = await res.json();
      if (topics.length && !topic_id) {
        topic_id = topics[0].id;
      }
    } catch (e) {
      errorTopics = e.message;
    } finally {
      loadingTopics = false;
    }
  }

  // 2) Если есть id, подгружаем данные слова
  async function loadWord() {
    if (!id) return;
    loadingWord = true;
    errorWord = null;
    try {
      const res = await fetch(`http://localhost:8000/words/${id}`, {
        headers: { Accept: 'application/json' }
      });
      if (res.status === 422) {
        const err = await res.json();
        throw new Error(`Ошибка валидации: ${JSON.stringify(err.detail)}`);
      }
      if (!res.ok) throw new Error(`Ошибка ${res.status}`);
      const data = await res.json();
      topic_id = data.topic_id;
      raw_text = data.raw_text;
      description_raw_text = data.descriptions.length
        ? data.descriptions[0].raw_text
        : '';
      language = data.language;
      info = data.info || '';
    } catch (e) {
      errorWord = e.message;
    } finally {
      loadingWord = false;
    }
  }

  // При старте страницы (mount) сначала темы, потом (если есть id) слово
  onMount(async () => {
    await loadTopics();
    if (id) {
      await loadWord();
    }
  });

  // 3) Обработка сабмита (POST или PUT)
  async function handleSubmit() {
    submitError = null;

    // Проверяем, что topic_id и raw_text непустые
    if (!topic_id || !raw_text.trim()) {
      submitError = 'Тема и само слово (raw_text) обязательны.';
      return;
    }

    submitting = true;
    try {
      const payload = {
        topic_id,
        raw_text: raw_text.trim(), // В режиме редактирования это не меняется
        description_raw_text: description_raw_text.trim(),
        language,
        info: info.trim()
      };

      let endpoint = 'http://localhost:8000/words/';
      let method = 'POST';
      if (id) {
        endpoint = `http://localhost:8000/words/${id}`;
        method = 'PUT';
      }

      const res = await fetch(endpoint, {
        method,
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (res.status === 422) {
        const errJson = await res.json();
        submitError = 'Ошибка валидации: ' + JSON.stringify(errJson.detail);
        return;
      }
      if (!res.ok) {
        // Выводим тело ошибки, если оно есть
        let text;
        try {
          const errJson = await res.json();
          text = JSON.stringify(errJson);
        } catch {
          text = await res.text();
        }
        throw new Error(`Ошибка ${res.status}: ${text}`);
      }

      const createdOrUpdated = await res.json();
      goto(`/word/${createdOrUpdated.id}`);
    } catch (e) {
      submitError = e.message;
    } finally {
      submitting = false;
    }
  }
</script>

<main>
  <h2>
    {#if id}
      Редактировать слово
    {:else}
      Добавить новое слово
    {/if}
  </h2>

  {#if loadingTopics}
    <p>Загрузка списка тем…</p>
  {:else if errorTopics}
    <p class="error">Ошибка загрузки тем: {errorTopics}</p>
  {:else if id && loadingWord}
    <p>Загрузка данных слова…</p>
  {:else if id && errorWord}
    <p class="error">Ошибка загрузки слова: {errorWord}</p>
  {:else}
    <form on:submit|preventDefault={handleSubmit}>
      <div class="field">
        <label for="topic">Тема</label>
        <select id="topic" bind:value={topic_id} required>
          {#each topics as t}
            <option value={t.id}>{t.name}</option>
          {/each}
        </select>
      </div>

      <div class="field">
        <label for="raw_text">Слово (raw_text)</label>
        <!-- Если id есть (режим редактирования), raw_text делаем readonly -->
        <input
          id="raw_text"
          type="text"
          bind:value={raw_text}
          placeholder="Введите слово"
          required
          {...(id ? { readonly: true } : {})}
        />
      </div>

      <div class="field">
        <label for="description_raw_text">Описание (description_raw_text)</label>
        <textarea
          id="description_raw_text"
          rows="3"
          bind:value={description_raw_text}
          placeholder="Описание (необязательно)"
        ></textarea>
      </div>

      <div class="field">
        <label for="language">Язык</label>
        <select id="language" bind:value={language}>
          <option value="russian">russian</option>
          <option value="english">english</option>
        </select>
      </div>

      <div class="field">
        <label for="info">Дополнительная информация (info)</label>
        <input
          id="info"
          type="text"
          bind:value={info}
          placeholder="Доп. информация (необязательно)"
        />
      </div>

      {#if submitError}
        <p class="error">{submitError}</p>
      {/if}

      <button type="submit" class="submit-button" disabled={submitting}>
        {#if submitting}
          {id ? 'Сохраняем изменения…' : 'Сохраняем…'}
        {:else}
          {id ? 'Обновить слово' : 'Сохранить'}
        {/if}
      </button>
    </form>
  {/if}
</main>

<style>
  main {
    padding: 1rem;
    max-width: 600px;
    margin: 0 auto;
  }
  h2 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .field {
    display: flex;
    flex-direction: column;
  }
  label {
    font-weight: bold;
    margin-bottom: 0.5rem;
  }
  input[type='text'],
  select,
  textarea {
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 1rem;
    box-sizing: border-box;
  }
  input[readonly] {
    background-color: #f5f5f5;
    color: #555;
  }
  textarea {
    resize: vertical;
  }
  .submit-button {
    width: fit-content;
    padding: 0.6rem 1.2rem;
    background-color: #0077cc;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  .submit-button:disabled {
    background-color: #aaa;
    cursor: default;
  }
  .submit-button:hover:not(:disabled) {
    background-color: #005fa3;
  }
  .error {
    color: #cc0000;
  }
</style>
