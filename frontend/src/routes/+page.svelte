<!-- src/routes/+page.svelte -->
<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Tabs from '$lib/components/Tabs.svelte';
  import SearchBar from '$lib/components/SearchBar.svelte';
  import WordList from '$lib/components/WordList.svelte';
  import { placeholder } from '$lib/stores/searchPlaceholder';

  let topics = [];
  let selectedTopicId = '';
  let selectedTopicName = '';
  let search = '';
  let words = [];
  let loadingTopics = false;
  let loadingWords = false;
  let errorTopics = null;
  let errorWords = null;
  let mounted = false;

  const RUSSIAN_ALPHABET = [
    'А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н',
    'О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я'
  ];

  let debounceTimeout;

  // Получаем список тем
  async function loadTopics() {
    loadingTopics = true;
    errorTopics = null;
    try {
      const res = await fetch('http://localhost:8000/topics/', {
        headers: { Accept: 'application/json' }
      });
      if (!res.ok) throw new Error(`Ошибка загрузки тем: ${res.status}`);
      const data = await res.json();
      topics = data;
      if (topics.length) {
        selectedTopicId = topics[0].id;
        selectedTopicName = topics[0].name;
      }
    } catch (e) {
      errorTopics = e.message;
    } finally {
      loadingTopics = false;
    }
  }

  // Загружаем слова: если search пустой, по 5 слов каждой буквы; иначе только по поиску
  async function loadWords() {
    if (!selectedTopicId) {
      words = [];
      return;
    }

    loadingWords = true;
    errorWords = null;
    words = [];

    try {
      if (search.trim()) {
        // Поиск: один запрос, без first_letter, возвращаем все совпадения
        const params = new URLSearchParams({
          topic_id: selectedTopicId,
          q: search.trim(),
          lang: 'russian',
          page: '1',
          size: '100'
        });
        const res = await fetch(`http://localhost:8000/words/?${params.toString()}`, {
          headers: { Accept: 'application/json' }
        });
        if (!res.ok) throw new Error(`Ошибка загрузки слов: ${res.status}`);
        const data = await res.json();
        words = data.map(item => ({
          id: item.id,
          term: item.cleaned_text || item.raw_text || ''
        }));
      } else {
        // Нет поиска: по 5 слов каждой буквы
        for (const letter of RUSSIAN_ALPHABET) {
          const params = new URLSearchParams({
            topic_id: selectedTopicId,
            first_letter: letter,
            q: '',
            lang: 'russian',
            page: '1',
            size: '5'
          });
          const res = await fetch(`http://localhost:8000/words/?${params.toString()}`, {
            headers: { Accept: 'application/json' }
          });
          if (!res.ok) throw new Error(`Ошибка загрузки слов: ${res.status}`);
          const data = await res.json();
          const portion = data.map(item => ({
            id: item.id,
            term: item.cleaned_text || item.raw_text || ''
          }));
          if (portion.length) {
            words.push(...portion);
          }
        }
      }
    } catch (e) {
      errorWords = e.message;
      words = [];
    } finally {
      loadingWords = false;
    }
  }

  onMount(async () => {
    mounted = true;
    await loadTopics();
    await loadWords();
  });

  function selectTab(name) {
    const topic = topics.find(t => t.name === name);
    if (topic) {
      selectedTopicId = topic.id;
      selectedTopicName = topic.name;
      search = '';
      loadWords();
    }
  }

  function onSearchInput(val) {
    search = val;
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      loadWords();
    }, 500);
  }

  function goAdd() {
    goto('/add');
  }
</script>

<main>
  {#if loadingTopics}
    <p>Загрузка тем…</p>
  {:else if errorTopics}
    <p class="error">Ошибка: {errorTopics}</p>
  {:else}
    <Tabs
      tabs={topics.map(t => t.name)}
      selected={selectedTopicName}
      onSelect={selectTab}
    />

    <div class="controls">
      <SearchBar
        bind:value={search}
        placeholder={$placeholder || 'Поиск...'}
        onInput={onSearchInput}
      />
      <button class="add-button" on:click={goAdd}>
        Добавить слово
      </button>
    </div>

    {#if loadingWords}
      <p>Загрузка слов…</p>
    {:else if errorWords}
      <p class="error">Ошибка: {errorWords}</p>
    {:else}
      <WordList {words} />
    {/if}
  {/if}
</main>

<style>
  main {
    padding: 1rem;
    max-width: 800px;
    margin: 0 auto;
  }
  .controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  .add-button {
    padding: 0.5rem 1rem;
    background-color: #0077cc;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  .add-button:hover {
    background-color: #005fa3;
  }
  .error {
    color: #cc0000;
  }
</style>
