<script>
  export let steps = [];
</script>

<div class="pipeline">
  {#each steps as step, i}
    <div class="step">{step}</div>
    {#if i < steps.length - 1}
      <div class="arrow">→</div>
    {/if}
  {/each}
</div>

<style>
  .pipeline {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
  }
  .step {
    padding: 0.5rem 1rem;
    background: #f0f0f0;
    border-radius: 4px;
    white-space: nowrap;
  }
  .arrow {
    font-size: 1.5rem;
    line-height: 1;
  }
</style>
