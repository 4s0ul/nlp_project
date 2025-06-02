<script>
  import { goto } from '$app/navigation';
  import { placeholder } from '$lib/stores/searchPlaceholder';

  function handleClick(name) {
    placeholder.set(name);
    goto('/');
  }
</script>

<nav class="header">
  <div class="creators">
    <button type="button" class="name" on:click={() => handleClick('Гаврин')}>
      гаврин
    </button>
    <span class="plus">+</span>
    <button type="button" class="name" on:click={() => handleClick('Конев')}>
      конев
    </button>
    <span class="plus">+</span>
    <button type="button" class="name" on:click={() => handleClick('Кузнецов')}>
      кузнецов
    </button>
    <span class="plus">+</span>
    <button type="button" class="name" on:click={() => handleClick('Панкратов')}>
      панкратов
    </button>
  </div>
  <div class="actions">
    <slot name="actions" />
  </div>
</nav>

<style>
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background-color: #f5f5f5;
  }
  .creators {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  button.name {
    background: none;
    border: none;
    padding: 0;
    font-weight: 700;
    font-size: 1.4rem;
    background: linear-gradient(45deg, #ff512f, #dd2476);
    background-clip: text;   
    color: transparent;
    transition: transform 0.2s;
    cursor: pointer;
  }
  button.name:hover {
    transform: scale(1.1);
  }
  .plus {
    font-size: 1.2rem;
    color: #555;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
</style>
