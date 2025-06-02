<script>
  export let neighbors = [];
</script>

<div class="neighbors">
  <h3>Похожие слова</h3>
  {#if neighbors.length === 0}
    <p>Нет похожих слов.</p>
  {:else}
    <ul>
      {#each neighbors as w}
        <li><a href={`/word/${w.id}`}>{w.term}</a></li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .neighbors {
    margin-top: 2rem;
  }
  .neighbors h3 {
    margin-bottom: 0.5rem;
  }
  ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  li {
    margin-bottom: 0.25rem;
  }
  a {
    text-decoration: none;
    color: #0077cc;
  }
  a:hover {
    text-decoration: underline;
  }
</style>
