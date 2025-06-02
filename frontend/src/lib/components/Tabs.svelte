<script>
  export let tabs = [];
  export let selected = null;
  export let onSelect = (tab) => {};
</script>

<nav class="tabs">
  {#each tabs as tab}
    <button
      class:selected={tab === selected}
      on:click={() => onSelect(tab)}>
      {tab}
    </button>
  {/each}
</nav>

<style>
  .tabs {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  button {
    background: none;
    border: none;
    padding: 0.5rem 1rem;
    cursor: pointer;
    font-size: 1rem;
  }
  button.selected {
    border-bottom: 2px solid #0077cc;
    font-weight: bold;
  }
</style>
