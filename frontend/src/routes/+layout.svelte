<script>
  import Header from '$lib/components/Header.svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';

  const goHome = () => goto('/');
</script>

<Header>
  <!-- Лого -->
  <a href="/" class="logo">
    Гаврин + Конев + Кузнецов + Панкратов
  </a>

  <!-- Actions: только Home на любых страницах, кроме главной -->
  <div slot="actions">
    {#if $page.url.pathname !== '/'}
      <!-- <button class="home-button" on:click={goHome}>
        Home
      </button> -->
    {/if}
  </div>
</Header>

<main>
  <slot />
</main>

<style>
  main {
    padding: 1rem;
    max-width: 800px;
    margin: 0 auto;
  }
  button.home-button {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    background-color: #0077cc;
    color: #fff;
    cursor: pointer;
    font-size: 1rem;
  }
  button.home-button:hover {
    background-color: #005fa3;
  }
</style>
