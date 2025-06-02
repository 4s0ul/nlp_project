<!-- src/lib/components/WordList.svelte -->
<script>
  export let words = [];

  // Русский алфавит в нужном порядке
  const RUSSIAN_ALPHABET = [
    'А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н',
    'О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я'
  ];

  // Группируем слова по первой букве (приводим букву к верхнему регистру)
  $: grouped = words.reduce((acc, w) => {
    const letter = w.term[0]?.toUpperCase() || '';
    // Берём только букву A–Я, всё остальное пропускаем
    if (RUSSIAN_ALPHABET.includes(letter)) {
      (acc[letter] ||= []).push(w);
    }
    return acc;
  }, {});

  // Формируем массив групп в порядке русского алфавита
  $: sortedGroups = RUSSIAN_ALPHABET.reduce((obj, letter) => {
    if (grouped[letter]) {
      // Сортируем слова внутри группы по алфавиту
      const sortedItems = grouped[letter].sort((a, b) =>
        a.term.localeCompare(b.term, 'ru')
      );
      // Берём максимум 5 слов
      obj[letter] = sortedItems.slice(0, 5);
    }
    return obj;
  }, {});
</script>

{#if words.length === 0}
  <p>Нет слов.</p>
{:else}
  {#each Object.entries(sortedGroups) as [letter, items]}
    <div class="group">
      <h3>{letter}</h3>
      <ul>
        {#each items as w}
          <li><a href={`/word/${w.id}`}>{w.term}</a></li>
        {/each}
      </ul>
    </div>
  {/each}
{/if}

<style>
  .group {
    margin-bottom: 1rem;
  }
  h3 {
    margin: 0.5rem 0;
    font-weight: bold;
  }
  ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  li {
    margin-left: 1rem;
  }
  a {
    text-decoration: none;
    color: #0077cc;
  }
  a:hover {
    text-decoration: underline;
  }
</style>
