<script>
  export let steps = [];
  const labels = ['Сырой текст', 'Очистка', 'Стемминг'];
</script>

<div class="pipeline-container">
  {#each steps as step, i}
    <div class="stage">
      <div class="box">{step}</div>
      <div class="label">{labels[i]}</div>
    </div>

    {#if i < steps.length - 1}
      <div class="arrow">→</div>
    {/if}
  {/each}
</div>

<style>

  .pipeline-container {
    display: flex;
    align-items: center;      
    justify-content: center;  
    flex-wrap: wrap;          
    gap: 0.75rem;             
    margin-bottom: 1rem;
  }


  .stage {
    display: flex;
    flex-direction: column;
    align-items: center;  
    max-width: 150px;      
  }

  .box {
    background-color: #fafafa;
    border: 1px solid #ddd;
    padding: 0.6rem 1rem;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #333;
    text-align: center;
    word-break: break-word; 
    min-width: 80px;       
  }

  .label {
    margin-top: 0.3rem;
    font-size: 0.75rem;
    color: #666;
    text-align: center;
  }

  .arrow {
    margin: 0 0.5rem;
    font-size: 1.2rem;
    color: #999;
    line-height: 1;
  }

  @media (max-width: 500px) {
    .box {
      font-size: 0.8rem;
      padding: 0.5rem 0.8rem;
      min-width: 60px;
    }
    .label {
      font-size: 0.7rem;
    }
    .arrow {
      font-size: 1rem;
      margin: 0 0.3rem;
    }
  }
</style>
