<!-- src/routes/word/[id]/+page.svelte -->
<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import PipelineViz from '$lib/components/PipelineViz.svelte';
  import NearestWords from '$lib/components/NearestWords.svelte';


  let id;
  $: id = $page.params.id;

  let detail = null;
  let loading = false;
  let error = null;

  // Похожие слова (neighbors)
  let similar = [];
  let loadingSimilar = false;
  let errorSimilar = null;

  // Ошибка при удалении (из предыдущей задачи)
  let deleteError = null;

  /**
   * Загружает данные о слове (GET /words/{id})
   */
  async function loadDetail() {
    if (!id) return;
    loading = true;
    error = null;
    detail = null;
    try {
      const res = await fetch(`http://localhost:8000/words/${id}`, {
        headers: { Accept: 'application/json' }
      });
      if (res.status === 422) {
        const err = await res.json();
        throw new Error(`Validation Error: ${JSON.stringify(err.detail)}`);
      }
      if (!res.ok) throw new Error(`Ошибка ${res.status}`);
      const data = await res.json();

      const pipeline = [
        data.raw_text,
        data.cleaned_text || data.raw_text,
        data.lemmatized_text || data.cleaned_text || data.raw_text
      ];
      const definition = data.descriptions.length
        ? data.descriptions[0].raw_text
        : '';

      detail = {
        id: data.id,
        term: data.cleaned_text || data.raw_text,
        definition,
        pipeline,
        // neighbors пока пустой, заполним отдельно
        neighbors: []
      };

      // Если у слова есть descriptions, подгружаем похожие
      if (data.descriptions.length) {
        await loadSimilar(data.descriptions[0].id);
      }
    } catch (e) {
      error = e.message;
      detail = null;
    } finally {
      loading = false;
    }
  }

  $: if (id) {
   loadDetail();
  }

  /**
   * Загружает похожие слова по description_id (GET /words/similar/?description_id=...&top_k=5)
   */
  async function loadSimilar(description_id, top_k = 5) {
    loadingSimilar = true;
    errorSimilar = null;
    similar = [];
    try {
      const params = new URLSearchParams({
        description_id,
        top_k: String(top_k)
      });
      // Добавили слэш после 'similar/'
      const res = await fetch(
        `http://localhost:8000/search/similar/?${params.toString()}`,
        { headers: { Accept: 'application/json' } }
      );
      if (res.status === 422) {
        const err = await res.json();
        throw new Error(`Validation Error: ${JSON.stringify(err.detail)}`);
      }
      if (!res.ok) throw new Error(`Ошибка ${res.status}`);
      const data = await res.json();
      similar = data.map(item => ({
        id: item.id,
        term: item.cleaned_text || item.raw_text || ''
      }));
      if (detail) detail.neighbors = similar;
    } catch (e) {
      errorSimilar = e.message;
      similar = [];
    } finally {
      loadingSimilar = false;
    }
  }

  onMount(loadDetail);

  function handleEdit() {
    goto(`/add?id=${id}`);
  }

  /**
   * Удаление через DELETE /words/{id}
   */
  async function handleDelete() {
    deleteError = null;
    if (!detail) return;
    if (!confirm(`Вы уверены, что хотите удалить слово «${detail.term}»?`)) {
      return;
    }
    try {
      const res = await fetch(`http://localhost:8000/words/${id}`, {
        method: 'DELETE',
        headers: { Accept: 'application/json' }
      });
      if (res.status === 422) {
        const err = await res.json();
        throw new Error(`Validation Error: ${JSON.stringify(err.detail)}`);
      }
      if (!res.ok) {
        let text;
        try {
          const errJson = await res.json();
          text = JSON.stringify(errJson);
        } catch {
          text = await res.text();
        }
        throw new Error(`Ошибка ${res.status}: ${text}`);
      }
      goto('/');
    } catch (e) {
      deleteError = e.message;
    }
  }
</script>

<main>
  {#if loading}
    <p>Загрузка…</p>
  {:else if error}
    <p class="error">Ошибка: {error}</p>
  {:else if detail}
    <h2 class="term">{detail.term}</h2>
    {#if detail.definition}
      <p class="definition">{detail.definition}</p>
    {/if}

    <PipelineViz steps={detail.pipeline} />

    <!-- Блок похожих слов -->
    {#if loadingSimilar}
      <p>Загрузка похожих слов…</p>
    {:else if errorSimilar}
      <p class="error">Ошибка при загрузке похожих слов: {errorSimilar}</p>
    {:else if similar.length}
      <section class="section">
        <h3>Похожие слова</h3>
        <NearestWords neighbors={similar} />
      </section>
    {/if}

    {#if deleteError}
      <p class="error">Ошибка при удалении: {deleteError}</p>
    {/if}

    <div class="actions">
      <button class="edit" on:click={handleEdit}>Редактировать</button>
      <button class="delete" on:click={handleDelete}>Удалить</button>
    </div>
  {:else}
    <p>Слово не найдено.</p>
  {/if}
</main>

<style>
  main {
    padding: 1rem;
    max-width: 700px;
    margin: 0 auto;
  }
  .term {
    font-size: 2rem;
    margin-bottom: 0.5rem;
  }
  .definition {
    margin: 0.5rem 0 1.5rem;
    font-size: 1.1rem;
    color: #333;
  }
  .section {
    margin: 1.5rem 0;
  }
  h3 {
    margin-bottom: 0.5rem;
  }
  .actions {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
  }
  .actions button {
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.2s;
  }
  .actions button.edit {
    background-color: #0077cc;
    color: #fff;
  }
  .actions button.edit:hover {
    background-color: #005fa3;
  }
  .actions button.delete {
    background-color: #cc0000;
    color: #fff;
  }
  .actions button.delete:hover {
    background-color: #a30000;
  }
  .error {
    color: #cc0000;
    margin-top: 1rem;
  }
</style>
