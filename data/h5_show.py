import h5py
import pandas as pd
import os

def interactive_h5_explorer(h5_file_obj, current_path):
    """
    HDF5 파일의 특정 경로를 대화형으로 탐색하고 사용자 입력을 처리합니다.

    Args:
        h5_file_obj: h5py.File 객체.
        current_path: 현재 탐색 중인 HDF5 경로 (예: '/', '/weather/', '/weather/block0_values').

    Returns:
        다음 경로 (문자열), 특별 명령 문자열 ('__EXIT__', '__SELECT_FILE__'), 또는 오류 시 None.
    """
    try:
        target_obj = h5_file_obj[current_path]
    except KeyError:
        print(f"오류: 경로 '{current_path}'을(를) 파일에서 찾을 수 없습니다.")
        return None # 예상치 못한 오류 발생 시 None 반환

    print(f"\n--- 현재 경로: {current_path} ---")

    # 현재 객체의 어트리뷰트 출력
    print(f"\n[어트리뷰트]:")
    if target_obj.attrs:
        for key, value in target_obj.attrs.items():
            # 바이트 어트리뷰트 처리
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore')
            print(f"  {key}: {value}")
    else:
        print("  (어트리뷰트 없음)")

    # 현재 객체가 그룹인 경우, 하위 항목 출력 및 선택지 제공
    if isinstance(target_obj, h5py.Group):
        print("\n[하위 항목]:")
        items = []
        if len(target_obj) == 0:
            print("  (하위 항목 없음)")
        else:
            for i, name in enumerate(target_obj):
                item = target_obj[name]
                item_type_display = ""
                if isinstance(item, h5py.Group):
                    item_type_display = "그룹"
                    display_name = f"{name}/"
                elif isinstance(item, h5py.Dataset):
                    item_type_display = "데이터셋"
                    display_name = f"{name} (형태: {item.shape}, 타입: {item.dtype})"
                else:
                    item_type_display = "알 수 없는 객체"
                    display_name = f"{name} (타입: {type(item)})"
                
                items.append((name, item_type_display)) # (실제 이름, 표시 타입)
                print(f"  {i+1}. {item_type_display}: {display_name}")
        
        print("\n[명령]:")
        print("  숫자를 입력하여 하위 항목으로 이동")
        print("  'q'를 입력하여 상위 그룹으로 이동 (최상위 그룹 '/'에서는 파일 선택으로 돌아감)")
        print("  'exit'를 입력하여 종료")
        
        while True:
            choice = input("선택: ").strip()
            if choice.lower() == 'q':
                if current_path == '/':
                    # 최상위 그룹에서 'q' 입력 시 파일 선택 화면으로 돌아갑니다.
                    print("최상위 그룹입니다. 파일 선택 화면으로 돌아갑니다.")
                    return '__SELECT_FILE__'
                else:
                    # 상위 경로로 이동 (예: '/group/subgroup' -> '/group')
                    parent_path = os.path.dirname(current_path)
                    if not parent_path: # os.path.dirname('/group')은 ''를 반환하므로, '/'로 처리
                        return '/'
                    return parent_path
            elif choice.lower() == 'exit':
                return '__EXIT__' # 종료 신호
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    selected_name = items[idx][0]
                    # 새 경로 구성. h5py 경로는 '/'로 구분됩니다.
                    if current_path == '/':
                        return f"/{selected_name}"
                    else:
                        return f"{current_path}/{selected_name}"
                else:
                    print("잘못된 번호입니다. 다시 입력해주세요.")
            except ValueError:
                print("잘못된 입력입니다. 숫자, 'q' 또는 'exit'를 입력해주세요.")

    # 현재 객체가 데이터셋인 경우, 데이터 정보 및 일부 데이터 출력
    elif isinstance(target_obj, h5py.Dataset):
        print(f"\n[데이터셋 정보]:")
        print(f"  형태 (Shape): {target_obj.shape}")
        print(f"  데이터 타입 (Dtype): {target_obj.dtype}")
        
        current_row_offset = 0
        rows_to_display = 10 # 초기 표시 행 수

        while True:
            print(f"\n[데이터셋 내용 (현재 {current_row_offset}부터 {min(current_row_offset + rows_to_display, target_obj.shape[0])}까지)]:")
            if target_obj.shape and target_obj.shape[0] > 0: # 데이터셋이 비어있지 않고 행이 있는 경우
                try:
                    total_rows = target_obj.shape[0]
                    end_row = min(current_row_offset + rows_to_display, total_rows)
                    
                    if current_row_offset >= total_rows:
                        print("  (더 이상 표시할 데이터가 없습니다. 'q' 또는 'exit'를 입력해주세요.)")
                    else:
                        data_slice = target_obj[current_row_offset:end_row]

                        if target_obj.dtype.kind == 'S': # 바이트 문자열인 경우 디코딩
                            if data_slice.ndim > 0: # 슬라이스가 비어있지 않은 경우
                                decoded_data = [d.decode('utf-8', errors='ignore') for d in data_slice]
                                for item in decoded_data:
                                    print(f"  {item}")
                            else:
                                print("  (데이터 슬라이스가 비어 있습니다)")
                        else:
                            if data_slice.ndim > 0: # 슬라이스가 비어있지 않은 경우
                                if data_slice.ndim > 1: # 2차원 이상 배열
                                    for row in data_slice:
                                        print(f"  {row}")
                                else: # 1차원 배열
                                    for item in data_slice:
                                        print(f"  {item}")
                            else:
                                print("  (데이터 슬라이스가 비어 있습니다)")
                        
                        if end_row < total_rows:
                            print(f"  ... (총 {total_rows}개 중 {end_row}개 표시됨)")
                        else:
                            print(f"  (모든 {total_rows}개 데이터 표시됨)")

                except Exception as data_read_error:
                    print(f"  데이터를 읽는 중 오류 발생: {data_read_error}")
                    print("  (이 데이터셋은 너무 크거나 복잡하여 일부만 표시할 수 없습니다.)")
            else:
                print("  (데이터셋이 비어 있습니다)")
        
            print("\n[명령]:")
            print("  'q'를 입력하여 상위 그룹으로 이동")
            print("  'n [숫자]'를 입력하여 다음 [숫자]개 데이터 표시 (예: 'n 20')")
            print("  'all'을 입력하여 모든 데이터 표시 (주의: 대용량 데이터는 느려질 수 있음)")
            print("  'exit'를 입력하여 종료")

            choice = input("선택: ").strip().lower()
            if choice == 'q':
                parent_path = os.path.dirname(current_path)
                if not parent_path: # '/dataset'과 같은 경우 ''를 반환하므로, '/'로 처리
                    return '/'
                return parent_path
            elif choice.startswith('n '):
                try:
                    num_rows = int(choice.split(' ')[1])
                    if num_rows > 0:
                        current_row_offset += num_rows
                    else:
                        print("유효한 숫자를 입력해주세요.")
                except ValueError:
                    print("잘못된 형식입니다. 'n [숫자]' 형식으로 입력해주세요 (예: 'n 20').")
            elif choice == 'all':
                print("\n경고: 모든 데이터를 로드합니다. 대용량 데이터셋의 경우 시간이 오래 걸리거나 메모리 문제가 발생할 수 있습니다.")
                current_row_offset = 0
                rows_to_display = target_obj.shape[0] if target_obj.shape else 0
                if rows_to_display == 0:
                    print("  (표시할 데이터가 없습니다)")
                    continue # 다시 프롬프트로
                # 'all' 선택 시 한 번에 모든 데이터를 출력하고 다시 프롬프트로 돌아감
                # 다음 루프에서 'q'나 'exit'를 기다리게 됨
                # 이 부분을 무한 루프에서 벗어나게 하려면 return을 추가해야 하지만,
                # 사용자에게 계속 옵션을 제공하기 위해 현재 상태 유지
            elif choice == 'exit':
                return '__EXIT__' # 종료 신호
            else:
                print("잘못된 입력입니다. 유효한 명령을 입력해주세요.")
    
    return current_path # 그룹 탐색 후 다음 경로를 반환 (이 부분은 그룹 탐색 루프에서만 사용됨)

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    while True: # 파일 선택 루프
        # 현재 디렉토리에서 .h5 파일 목록을 찾습니다.
        h5_files = [f for f in os.listdir('.') if f.endswith('.h5') and os.path.isfile(f)]

        if not h5_files:
            print("오류: 현재 디렉토리에서 '.h5' 파일을 찾을 수 없습니다.")
            print("탐색할 '.h5' 파일을 현재 디렉토리에 넣어주세요.")
            input("아무 키나 눌러 다시 시도하거나 종료하려면 Ctrl+C를 누르세요...")
            continue # 파일이 없으면 다시 파일 검색 루프 시작

        print("\n--- 탐색할 HDF5 파일을 선택해주세요 ---")
        for i, filename in enumerate(h5_files):
            print(f"  {i+1}. {filename}")
        print("  'exit'를 입력하여 종료")

        selected_h5_filename = None
        while selected_h5_filename is None:
            choice = input("파일 번호 선택: ").strip().lower()
            if choice == 'exit':
                print("HDF5 파일 탐색을 종료합니다.")
                exit() # 프로그램 종료
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(h5_files):
                    selected_h5_filename = h5_files[idx]
                else:
                    print("잘못된 번호입니다. 다시 입력해주세요.")
            except ValueError:
                print("잘못된 입력입니다. 숫자 또는 'exit'를 입력해주세요.")

        print(f"\n선택된 파일: '{selected_h5_filename}'")
        current_h5_path = '/' # 탐색 시작 경로 (최상위 그룹)
        
        # 선택된 HDF5 파일을 한 번 열고, 파일 객체를 계속 사용합니다.
        with h5py.File(selected_h5_filename, 'r') as f:
            while True: # 파일 내부 탐색 루프
                # 대화형 탐색 함수 호출
                next_path = interactive_h5_explorer(f, current_h5_path)
                
                if next_path == '__EXIT__': # 'exit' 명령으로 종료 신호가 오면
                    print("\nHDF5 파일 탐색을 종료합니다.")
                    exit() # 프로그램 종료
                elif next_path == '__SELECT_FILE__': # 최상위 그룹에서 'q' 입력 시 파일 선택으로 돌아감
                    break # 현재 파일 탐색 루프를 벗어나 파일 선택 루프로 돌아감
                elif next_path is None: # 예상치 못한 오류로 인한 종료
                    print("\n예상치 못한 오류로 HDF5 파일 탐색을 종료합니다.")
                    exit()
                current_h5_path = next_path # 다음 경로로 업데이트
