<script>
  export let value = '';
  export let placeholder = 'Search...';
  export let onInput = (val) => {};
  function handleInput(e) {
    onInput(e.target.value);
  }
</script>

<input
  type="text"
  bind:value
  placeholder={placeholder}
  on:input={handleInput}
/>

<style>
  input {
    flex: 1;
    padding: 0.5rem;
    font-size: 1rem;
    border: 1px solid #ccc;
    border-radius: 4px;
  }

  input::placeholder {
    color: #ff6b6b;
    opacity: 0.8;
  }

  input:focus::placeholder {
    color: #ff4757;   
    opacity: 0.8;
  }
</style>
